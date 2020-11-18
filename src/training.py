import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import get_fixed_temperature, get_losses, create_logger
from torch.utils.tensorboard import SummaryWriter
from generator import *
from discriminator import *
import numpy as np
from torch.utils.data import DataLoader
from tasks import *
from metrics.perplexity import calc_pp
import nltk
from nltk.translate.bleu_score import SmoothingFunction

class GANInstructor():
    def __init__(self, args, train_dataset, dev_dataset):

        # generator, discriminator
        self.gen = Generator(args).to(args.device)
        self.disc = Discriminator(args).to(args.device)
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.log_file + ".txt")
        # Optimizer
        self.pretrain_opt = optim.Adam(self.gen.parameters(), lr=args.pretrain_lr)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=args.gen_lr)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=args.disc_lr)

        self.pre_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.pre_train_batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
        self.pre_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.pre_eval_batch_size, num_workers=args.num_workers, collate_fn=dev_dataset.collate_fn)

        self.adv_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.adv_train_batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
        self.adv_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.adv_eval_batch_size, num_workers=args.num_workers, collate_fn=dev_dataset.collate_fn)
       
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
 
        self.model_dir = args.model_dir
        self.writer = SummaryWriter(args.save_dir)
        
        self.pretrain_steps = 0
        self.gen_steps = 0
        self.disc_steps = 0

        self.args = args
        self.cgan = (self.args.conditional_gan==1)
        self.adv_epoch = -1
        self.pretrain_patience = self.args.pretrain_patience
        self.advtrain_patience = self.args.advtrain_patience

    def genpretrain_loop(self, what):

        gen_loss = []
        criterion = nn.CrossEntropyLoss(ignore_index=1)
        all_references = []
        all_candidates = []
        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total = (len(self.train_dataset) if what=='train' else len(self.dev_dataset))) as progress:
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.pre_train_loader if what=='train' else self.pre_eval_loader)):
                
                r_captions = self.train_dataset.convert_to_tokens_references(captions['input_ids'])
                all_references += r_captions

                images = images.to(self.args.device) 
                lengths = lengths.to(self.args.device)
                attn_mask = captions["attention_mask"].to(self.args.device)
                captions = captions["input_ids"].to(self.args.device)
                real_captions = captions        
                
                if self.cgan:
                    features = [self.gen.encoder(images),self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
                else:
                    features = self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
                gen_captions, gen_caption_ids = self.gen.decoder.sample(features, pretrain=True, max_caption_len=max_caption_len)
                real_captions, gen_captions = real_captions.to(self.args.device), gen_captions.to(self.args.device)

                g_captions = self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)
                all_candidates += g_captions             
    
                loss = criterion(gen_captions.view(-1,gen_captions.size(-1)), real_captions.view(-1))

                if what == 'train':
                    self.optimize(self.pretrain_opt, loss, self.gen)

                gen_loss.append(loss.item())

                self.writer.add_scalar('GenPreTraining_train_loss' if what=='train' else 'GenPreTraining_val_loss',loss,self.pretrain_steps)         
                progress.update(len(images))
                progress.set_postfix(loss=loss.item())        
        
        # print(all_references[-10:], all_candidates[-10:])
        return (gen_loss , all_references, all_candidates)

    def pretrain_generator(self, epochs , weights=[0.25,0.25,0.25,0.25]):
        self.log.info("Pretraining Generator")
        total_loss = 0

        best_loss = None
        patience = 0
        for epoch in range(self.args.pretrain_epochs):

            
            self.gen.train()
            gen_loss, ref, gen = self.genpretrain_loop('train')
            train_epoch_loss = np.mean(gen_loss)       

            train_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram -> BLEU-4
            total_loss += train_epoch_loss 

            self.gen.eval()
            gen_loss, ref, gen = self.genpretrain_loop('val')
            val_epoch_loss = np.mean(gen_loss)

            val_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram -> BLEU-4

            if epoch%self.args.pre_log_step == 0:
                train_perplexity = np.exp(train_epoch_loss)
                val_perplexity = np.exp(val_epoch_loss)
                self.log.info("Epoch {}: \n \t Train Loss: {} \n\t Val Loss: {} \n\t Train PP: {} \n\t Val PP: {} \n\t Train BLEU: {} \n\t Val BLEU: {} " \
                                .format(epoch,train_epoch_loss,val_epoch_loss, train_perplexity, val_perplexity,train_bleu, val_bleu))

            if best_loss is None or val_epoch_loss < best_loss :
                best_loss = val_epoch_loss
                torch.save(self.gen.state_dict(), self.args.model_dir + "/pretrained_model.ckpt")
                patience = 0
            elif patience >= self.pretrain_patience:
                self.log.info("Early Stopping at Epoch {}".format(epoch))
                break
            else:
                patience += 1


            self.pretrain_steps+=1

        return (total_loss/epochs if epochs!=0 else 0)
    
    def adv_loop(self, what):
        total_gen_loss = 0
        total_disc_loss = 0
        
        float_epoch = 0.0
        all_references = []
        all_candidates = []
        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=(len(self.train_dataset) if what == 'train' else len(self.dev_dataset))) as progress:
            gen_loss = []
            disc_loss = []
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.adv_train_loader if what=='train' else self.adv_eval_loader)):
                
                float_epoch += 1

                r_captions = self.train_dataset.convert_to_tokens_references(captions['input_ids'])
                all_references += r_captions

                images,lengths = images.to(self.args.device), lengths.to(self.args.device)
                attn_mask = captions["attention_mask"].to(self.args.device)
                captions = captions["input_ids"].to(self.args.device)

                real_captions = captions 
   
                if self.cgan:
                    features = [self.gen.encoder(images),self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
                else:
                    features = self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
       
                gen_captions, gen_caption_ids = self.gen.decoder.sample(features, max_caption_len=max_caption_len)
                fake_captions = gen_captions.detach()
                fake_captions = fake_captions.to(self.args.device)

                g_captions = self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)
                all_candidates += g_captions

                real_captions = F.one_hot(real_captions, self.args.vocab_size).float()

                # ===Train===
                d_out_real = self.disc(real_captions)
                d_out_fake = self.disc(fake_captions)
                g_out = self.disc(gen_captions)
                g_loss, d_loss = get_losses(d_out_real, d_out_fake, g_out, self.args.adv_loss_type)

                if what == 'train':
                    self.optimize(self.disc_opt, d_loss, self.disc, True)
                    self.optimize(self.gen_opt, g_loss, self.gen)

                self.writer.add_scalar('Discriminator_train_loss' if what=='train' else 'Discriminator_val_loss',d_loss,self.disc_steps)
                self.disc_steps+=1

                self.writer.add_scalar('Generator_train_loss' if what=='train' else 'Generator_val_loss',g_loss,self.gen_steps)
                self.gen_steps+=1

                gen_loss.append(g_loss.item())
                disc_loss.append(d_loss.item())

                progress.update(len(images))
                progress.set_postfix(disc_loss=d_loss.item(), gen_loss=g_loss.item())

                self.update_temperature(self.adv_epoch + (float_epoch/(len((self.adv_train_loader if what=='train' else self.adv_eval_loader)))), self.args.adv_epochs)  # update temperature

        total_gen_loss = np.mean(gen_loss)
        total_disc_loss = np.mean(disc_loss)

        return (total_gen_loss, total_disc_loss, all_references, all_candidates)

    def update_temperature(self, i, N):
        self.gen.decoder.temperature = get_fixed_temperature(self.args.temperature, i, N, self.args.temp_adpt)

    #@staticmethod
    def optimize(self, opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm)
        opt.step()

    def _run(self, weights=[0.25,0.25,0.25,0.25]):
    
        ## === PRETRAINING GENERATOR === ##
        self.pretrain_generator(self.args.pretrain_epochs, weights)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        
        patience = 0
        best_loss = None
        for adv_epoch in range(self.args.adv_epochs):

            self.adv_epoch = adv_epoch
            
            self.disc.train()
            self.gen.train()
            train_g_loss, train_d_loss, ref, gen = self.adv_loop('train')  # Discriminator
            train_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram BLEU-4
            
            self.disc.eval()
            self.gen.eval()
            val_g_loss, val_d_loss, ref, gen= self.adv_loop('val')
            val_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram BLEU-4
            # TEST
            if adv_epoch % self.args.adv_log_step == 0 or adv_epoch == self.args.adv_epochs - 1:
                val_perplexity = np.exp(val_g_loss)
                train_perplexity = np.exp(train_g_loss)
                self.log.info('[ADV] epoch %d (temperature: %.4f):\n\t g_loss: %.4f | %.4f \n\t d_loss: %.4f | %.4f \n\t Train PP: %.4f \n\t Val PP: %.4f \n\t Train BLEU: %.4f \n\t Val BLEU: %.4f' %\
                              (adv_epoch, self.gen.decoder.temperature, train_g_loss, val_g_loss, train_d_loss, val_d_loss, train_perplexity, val_perplexity, train_bleu, val_bleu))

            if best_loss is None or val_g_loss < best_loss :
                best_loss = val_g_loss 
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model.ckpt")
                patience = 0
            elif patience >= self.advtrain_patience:
                self.log.info("Early Stopping at Epoch {}".format(epoch))
                break
            else:
                patience += 1
