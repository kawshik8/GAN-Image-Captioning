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
import sys
sys.stdout.flush()

class GANInstructor():
    def __init__(self, args, train_dataset, dev_dataset):

        # generator, discriminator
        self.gen = Generator(args).to(args.device)
        self.disc = Discriminator(args).to(args.device)
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.log_file + ".txt")

        self.sent_log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.sent_log_file + ".txt")

        # Optimizer
        self.pretrain_opt = optim.Adam(self.gen.parameters(), lr=args.pretrain_lr)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=args.gen_lr)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=args.disc_lr)

        self.tokenizer = train_dataset.tokenizer

        #Schedulers ReduceLROnPlateau
        if args.gen_model_type == 'lstm':
            self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.pretrain_opt, patience=args.pretrain_lr_patience, verbose=True)
            self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.gen_opt, patience=args.gen_lr_patience, verbose=True)
            self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, patience=args.disc_lr_patience, verbose=True)
        else:

            self.pretrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.pretrain_opt, max_lr=1e-2, total_steps = args.pretrain_epochs*((len(train_dataset)//args.pre_train_batch_size)+1), final_div_factor = 4, pct_start=5/args.pretrain_epochs, anneal_strategy='cos')
            self.gen_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.gen_opt, max_lr=1e-3, total_steps = args.adv_epochs*((len(train_dataset)//args.adv_train_batch_size)+1), pct_start=5/args.pretrain_epochs, final_div_factor = 4, anneal_strategy='cos')
            self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, patience=args.disc_lr_patience, verbose=True)
        
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

        self.num_log_sent = 25

        self.teacher_force_choice_pre = 1.0
        self.teacher_force_choice_adv = 0.0

    def genpretrain_loop(self, what):

        gen_loss = []
        criterion = nn.CrossEntropyLoss(ignore_index=1)
        all_references = []
        all_candidates = []
        num_sent = 0

        with open('./sentences.txt', 'w') as f:

            with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total = (len(self.train_dataset) if what=='train' else len(self.dev_dataset))) as progress:
                for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.pre_train_loader if what=='train' else self.pre_eval_loader)):
                    
                    self.pretrain_opt.zero_grad()
                    r_captions = self.train_dataset.convert_to_tokens_references(captions['input_ids'])

                    all_references += r_captions


                    images = images.to(self.args.device) 
                    lengths = lengths.cpu()

                    attn_mask = captions["attention_mask"].to(self.args.device)
                    #s = torch.randint(10,(1,))
                    captions = captions["input_ids"].to(self.args.device)
                    real_captions = captions        
                    
                    if self.cgan:
                        features = self.gen.encoder(images)
                    else:
                        features = None

                    if torch.rand(1) < self.teacher_force_choice_pre:
                        gen_captions, gen_caption_ids = self.gen.decoder(features, captions, lengths, pretrain = True, attn_mask=attn_mask, max_caption_len = max_caption_len)
                    else:
                        gen_captions, gen_caption_ids = self.gen.decoder.sample(features, pretrain = True, max_caption_len = max_caption_len)
                    
                    real_captions, gen_captions = real_captions.to(self.args.device), gen_captions.to(self.args.device)

                    g_captions = self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)
                    all_candidates += g_captions             
        
                    if not self.args.conditional_gan:
                       real_captions = real_captions[:,1:]

                    f.write('\nReal Caption: {}'.format(self.train_dataset.convert_to_tokens_references(captions,skip_special_tokens = False)))
                    f.write('\nGenerated Caption: {}'.format(self.train_dataset.convert_to_tokens_candidates(gen_caption_ids,skip_special_tokens = False)))
                    #print(real_captions)
                    #print(gen_captions.max(dim=-1)[1])
                    # print(criterion(gen_captions.reshape(-1,gen_captions.size(-1)), real_captions.reshape(-1)).shape)

                    #losses = []
                    #accs = []
                    #for i in range(len(images)):
                    #    print(criterion(gen_captions[i], real_captions[i]))
                    #    losses.append(criterion(gen_captions[i], real_captions[i]))
                    #    accs.append(torch.equal(gen_captions[i].max(dim=-1)[1], real_captions[i]))
                    #    print(gen_captions[i].max(dim=-1)[1], real_captions[i])

                    #loss = torch.stack(losses).mean()
                    #print(accs)

                    loss = criterion(gen_captions.reshape(-1,gen_captions.size(-1)), real_captions.reshape(-1))
                    gen_loss.append(loss.item())

                    if what=='train':
                        loss.backward()
                        self.pretrain_opt.step()
                        if self.args.gen_model_type == 'transformer':
                            self.pretrain_scheduler.step()
                            print(self.pretrain_opt.param_groups[0]['lr'])
                    # if what == 'train':
                    #     self.optimize(self.pretrain_opt, loss, self.gen)

                    # total_norm = 0.0
                    # for p in self.gen.parameters():   
                    #     # print(p) 
                    #     param_norm = p.grad.data.norm(2)
                    #     total_norm += param_norm.item() ** 2
                    # total_norm = total_norm ** (1. / 2)
                    # print(total_norm)

                    self.writer.add_scalar('GenPreTraining_train_loss' if what=='train' else 'GenPreTraining_val_loss',loss,self.pretrain_steps)         
                    progress.update(len(images))
                    progress.set_postfix(loss=loss.item())#,norm=total_norm)        
        
        for i in range(10):
            
            self.sent_log.info("True Sentence : {} \nPred Sentence : {} \n".format(all_references[i],all_candidates[i]))

        # print(all_references[-10:], all_candidates[-10:])
        return (gen_loss , all_references, all_candidates)

    def pretrain_generator(self, epochs , weights=[0.25,0.25,0.25,0.25]):
        self.log.info("Pretraining Generator")
        self.sent_log.info("Pretraining Generator")
        total_loss = 0

        best_loss = None
        patience = 0

        for epoch in range(self.args.pretrain_epochs):

            self.sent_log.info("\nEpoch {}:".format(epoch))
            self.gen.train()
            gen_loss, ref, gen = self.genpretrain_loop('train')
            train_epoch_loss = np.mean(gen_loss)       

            train_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram -> BLEU-4
            total_loss += train_epoch_loss 

            self.gen.eval()
            gen_loss, ref, gen = self.genpretrain_loop('val')
            val_epoch_loss = np.mean(gen_loss)

            if self.args.gen_model_type == 'lstm':
                self.pretrain_scheduler.step(val_epoch_loss)

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
        num_sent = 0

        bce_loss = nn.BCEWithLogitsLoss()
        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=(len(self.train_dataset) if what == 'train' else len(self.dev_dataset))) as progress:
            gen_loss = []
            disc_loss = []
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.adv_train_loader if what=='train' else self.adv_eval_loader)):
                
                float_epoch += 1

                r_captions = self.train_dataset.convert_to_tokens_references(captions['input_ids'])
                all_references += r_captions

                images,lengths = images.to(self.args.device), lengths.cpu()
                attn_mask = captions["attention_mask"].to(self.args.device)
                captions = captions["input_ids"].to(self.args.device)

                real_captions = captions 
   
                if self.cgan:
                    features = [self.gen.encoder(images),self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
                else:
                    features = self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
       
                if torch.rand(1) < self.teacher_force_choice_pre:
                    gen_captions, gen_caption_ids = self.gen.decoder(features, captions, lengths, attn_mask=attn_mask, max_caption_len = max_caption_len)
                else:
                    gen_captions, gen_caption_ids = self.gen.decoder.sample(features, max_caption_len = max_caption_len)


                fake_captions = gen_captions.detach()
                fake_captions = fake_captions.to(self.args.device)

                #if num_sent < self.num_log_sent:
                #    num_sent += 1
                #    print(captions["input_ids"][0])
                #    print(gen_caption_ids[0])
                #    self.sent_log.info("True Sentence : {} \n Pred Sentence : {} \n".format(self.tokenizer.decode(captions["input_ids"][0]), self.tokenizer.decode(gen_caption_ids[0])))

                g_captions = self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)
                all_candidates += g_captions

                real_captions = F.one_hot(real_captions, self.args.vocab_size).float()

                # ===Train===
                d_out_real = self.disc(real_captions)
                d_out_fake = self.disc(fake_captions)

                d_loss_real = bce_loss(d_out_real, torch.ones_like(d_out_real))
                d_loss_fake = bce_loss(d_out_fake, torch.zeros_like(d_out_fake))
                d_loss = d_loss_real + d_loss_fake

                if what == 'train':
                    self.optimize(self.disc_opt, d_loss, self.disc, True)

                g_out = self.disc(gen_captions)
                g_loss = bce_loss(g_out, torch.ones_like(g_out))

                if what == 'train':
                    self.optimize(self.gen_opt, g_loss, self.gen)
                    if self.args.gen_model_type == 'transformer':
                        self.gen_scheduler.step()


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

        for i in range(10):
            self.sent_log.info("True Sentence : {} \n Pred Sentence : {} \n".format(all_references[i],all_candidates[i]))

        return (total_gen_loss, total_disc_loss, all_references, all_candidates)

    def update_temperature(self, i, N):
        self.gen.decoder.temperature = get_fixed_temperature(self.args.temperature, i, N, self.args.temp_adpt)

    #@staticmethod
    def optimize(self, opt, loss, model=None, retain_graph=False):
        
        loss.backward(retain_graph=retain_graph)
        # if model is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm)
        opt.step()

    def _run(self, weights=[0.25,0.25,0.25,0.25]):
    
        ## === PRETRAINING GENERATOR === ##
        self.pretrain_generator(self.args.pretrain_epochs, weights)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.sent_log.info('Starting Adversarial Training...')
        
        patience = 0
        best_loss = None
        for adv_epoch in range(self.args.adv_epochs):

            self.adv_epoch = adv_epoch

            self.sent_log.info("\nEpoch : {}".format(adv_epoch))
            
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

            if self.args.gen_model_type == 'lstm':
                self.gen_scheduler.step(val_g_loss)
            self.disc_scheduler.step(val_d_loss)

            if best_loss is None or val_g_loss < best_loss :
                best_loss = val_g_loss 
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model.ckpt")
                patience = 0
            elif patience >= self.advtrain_patience:
                self.log.info("Early Stopping at Epoch {}".format(adv_epoch))
                break
            else:
                patience += 1
