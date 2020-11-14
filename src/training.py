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
#from tasks import collate_fn

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

        self.pre_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.pre_train_batch_size, num_workers=4)
        self.pre_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.pre_eval_batch_size, num_workers=4)

        self.adv_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.adv_train_batch_size, num_workers=4)
        self.adv_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.adv_eval_batch_size, num_workers=4)
       
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

        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total = (len(self.train_dataset) if what=='train' else len(self.dev_dataset))) as progress:
            for batch_idx, (images, captions, lengths) in enumerate((self.pre_train_loader if what=='train' else self.pre_eval_loader)):
                
#                 print((self.pre_train_loader if what=='train' else self.pre_eval_loader).dataset.vocab_size, flush=True)
#                 print(captions, flush=True)
#                 print(lengths, flush=True)
                
                images,lengths = images.to(self.args.device), lengths.to(self.args.device)
                attn_mask = captions["attention_mask"].to(self.args.device)
                captions = captions["input_ids"].to(self.args.device)
                #attn_mask = captions["attention_mask"].to(self.args.device)

                real_captions = captions 
                # self.pretrain_opt.zero_grad()

                #images,captions,lengths = images.to(self.args.device), captions.to(self.args.device), lengths.to(self.args.device)              
                
                if self.cgan:
                    features = [self.gen.encoder(images),self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
                else:
                    features = self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
#                 fake_captions = self.gen.decoder.sample(features)

                # print(features.shape)
         
                gen_captions, gen_caption_ids = self.gen.decoder.sample(features, pretrain=True, max_caption_len=self.args.max_seq_len)

                real_captions, gen_captions = real_captions.to(self.args.device), gen_captions.to(self.args.device)

                # bleu = bleu_score(gen_caption_ids, real_captions.unsqueeze(0))
                # print(bleu)
#                 print(gen_captions.shape, real_captions.shape)
#                 targets = pack_padded_sequence(real_captions, lengths, batch_first=True, enforce_sorted=False)[0]
#                 outputs = pack_padded_sequence(gen_captions, lengths, batch_first=True, enforce_sorted=False)[0]      

                criterion = nn.CrossEntropyLoss()
    
                loss = criterion(gen_captions.view(-1,gen_captions.size(-1)), real_captions.view(-1))

                #print(loss)

                if what == 'train':
                    self.optimize(self.pretrain_opt, loss, self.gen)

                gen_loss.append(loss.item())

                self.writer.add_scalar('GenPreTraining_train_loss' if what=='train' else 'GenPreTraining_val_loss',loss,self.pretrain_steps)
                
                progress.update(len(images))
                progress.set_postfix(loss=loss.item())        
        
        return gen_loss

    def pretrain_generator(self, epochs):
        self.log.info("Pretraining Generator")
        total_loss = 0

        best_loss = None
        patience = 0
        for epoch in range(self.args.pretrain_epochs):

            self.gen.train()
            gen_loss = self.genpretrain_loop('train')
            train_epoch_loss = np.mean(gen_loss)       

            total_loss += train_epoch_loss 

            self.gen.eval()
            gen_loss = self.genpretrain_loop('val')
            val_epoch_loss = np.mean(gen_loss)

            if epoch%self.args.pre_log_step == 0:
                self.log.info("Epoch {}: \n \t Train: {} \n\t Val: {} ".format(epoch,train_epoch_loss,val_epoch_loss))

            if best_loss is None or val_epoch_loss < best_loss :
                best_loss = val_epoch_loss
                torch.save(self.gen.state_dict(), args.model_dir + "/pretrained_model.ckpt")
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
        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=(len(self.train_dataset) if what == 'train' else len(self.dev_dataset))) as progress:
            gen_loss = []
            disc_loss = []
            for batch_idx, (images, captions, lengths) in enumerate((self.adv_train_loader if what=='train' else self.adv_eval_loader)):
                
                float_epoch += 1
                
                images,lengths = images.to(self.args.device), lengths.to(self.args.device)
                attn_mask = captions["attention_mask"].to(self.args.device)
                captions = captions["input_ids"].to(self.args.device)
                #attn_mask = captions["attention_mask"].to(self.args.device)
                #images,captions,lengths = images.to(self.args.device), captions.to(self.args.device), lengths.to(self.args.device)
                real_captions = captions #train_data -> (images,lengths,captions)
          
                # self.disc_opt.zero_grad()
                # self.gen_opt.zero_grad()
                if self.cgan:
                    features = [self.gen.encoder(images),self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
                else:
                    features = self.gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
#                 fake_captions = self.gen.decoder.sample(features)
         
                gen_captions, gen_caption_ids = self.gen.decoder.sample(features, max_caption_len=self.args.max_seq_len)
                fake_captions = gen_captions.detach()

                fake_captions = fake_captions.to(self.args.device)

                # bleu = bleu_score(gen_caption_ids, real_captions.unsqueeze(0))
                # print(bleu)

                real_captions = F.one_hot(real_captions, self.args.vocab_size).float()
#                 fake_captions = F.one_hot(fake_captions, self.args.vocab_size).float()

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

        return (total_gen_loss, total_disc_loss)

    def update_temperature(self, i, N):
        self.gen.decoder.temperature = get_fixed_temperature(self.args.temperature, i, N, self.args.temp_adpt)

    #@staticmethod
    def optimize(self, opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm)
        opt.step()

    def _run(self):
    
        ## === PRETRAINING GENERATOR === ##
        self.pretrain_generator(self.args.pretrain_epochs)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
#         progress = tqdm(range(self.args.adv_epochs))
        
        patience = 0
        best_loss = None
        for adv_epoch in range(self.args.adv_epochs):

            self.adv_epoch = adv_epoch
            
            self.disc.train()
            self.gen.train()
            train_g_loss, train_d_loss = self.adv_loop('train')  # Discriminator
            
            self.disc.eval()
            self.gen.eval()
            val_g_loss, val_d_loss = self.adv_loop('val')

            # TEST
            if adv_epoch % self.args.adv_log_step == 0 or adv_epoch == self.args.adv_epochs - 1:
                self.log.info('[ADV] epoch %d (temperature: %.4f):\n\t g_loss: %.4f | %.4f \n\t d_loss: %.4f | %.4f' % (
                    adv_epoch, self.gen.decoder.temperature, train_g_loss, val_g_loss, train_d_loss, val_d_loss))

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
    
            

                # if cfg.if_save and not cfg.if_test:
                #     self._save('ADV', adv_epoch)
