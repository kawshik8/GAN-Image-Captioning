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
from tasks import collate_fn

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

        self.pre_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.pre_train_batch_size, collate_fn=collate_fn, num_workers=4)
        self.pre_eval_loader = DataLoader(dev_dataset, batch_size=args.pre_eval_batch_size, collate_fn=collate_fn, num_workers=4)

        self.adv_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.adv_train_batch_size, collate_fn=collate_fn, num_workers=4)
        self.adv_eval_loader = DataLoader(dev_dataset, batch_size=args.adv_eval_batch_size, collate_fn=collate_fn, num_workers=4)
       
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
 
        self.writer = SummaryWriter()
        
        self.pretrain_steps = 0
        self.gen_steps = 0
        self.disc_steps = 0

        self.args = args
        self.cgan = (self.args.conditional_gan==1)

    # def __del__(self):
    #     self.writer.close()

    def _run(self):

        ## === PRETRAINING GENERATOR === ##
        self.pretrain_generator(self.args.pretrain_epochs)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
#         progress = tqdm(range(self.args.adv_epochs))

        for adv_epoch in range(self.args.adv_epochs):
            self.disc.train()
            self.gen.train()
            train_g_loss, train_d_loss = self.adv_loop('train')  # Discriminator
            
            self.disc.eval()
            self.gen.eval()
            val_g_loss, val_d_loss = self.adv_loop('val')

            self.update_temperature(adv_epoch, self.args.adv_epochs)  # update temperature
    
            # TEST
            if adv_epoch % self.args.adv_log_step == 0 or adv_epoch == self.args.adv_epochs - 1:
                self.log.info('[ADV] epoch %d (temperature: %.4f):\n\t g_loss: %.4f | %.4f \n\t d_loss: %.4f | %.4f' % (
                    adv_epoch, self.gen.decoder.temperature, train_g_loss, val_g_loss, train_d_loss, val_d_loss))

                # if cfg.if_save and not cfg.if_test:
                #     self._save('ADV', adv_epoch)

    def genpretrain_loop(self, what):

        gen_loss = []

        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total = (len(self.train_dataset) if what=='train' else len(self.dev_dataset))) as progress:
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.pre_train_loader if what=='train' else self.pre_eval_loader)):
                
#                 print((self.pre_train_loader if what=='train' else self.pre_eval_loader).dataset.vocab_size, flush=True)
#                 print(captions, flush=True)
#                 print(lengths, flush=True)
                
                images,captions,lengths = images.to(self.args.device), captions.to(self.args.device), lengths.to(self.args.device)
                real_samples = captions 
                # self.pretrain_opt.zero_grad()

                #images,captions,lengths = images.to(self.args.device), captions.to(self.args.device), lengths.to(self.args.device)              
                
                if self.cgan:
                    features = self.gen.encoder(images)
#                 fake_samples = self.gen.decoder.sample(features)
         
                gen_samples, _ = self.gen.decoder.sample(features, pretrain=True, max_caption_len=max_caption_len)

                real_samples, gen_samples = real_samples.to(self.args.device), gen_samples.to(self.args.device)
#                 print(gen_samples.shape, real_samples.shape)
#                 targets = pack_padded_sequence(real_samples, lengths, batch_first=True, enforce_sorted=False)[0]
#                 outputs = pack_padded_sequence(gen_samples, lengths, batch_first=True, enforce_sorted=False)[0]      

                criterion = nn.CrossEntropyLoss()
    
                loss = criterion(gen_samples.view(-1,gen_samples.size(-1)), real_samples.view(-1))

                #print(loss)

                if what == 'train':
                    self.optimize(self.pretrain_opt, loss, self.gen)

                gen_loss.append(loss.item())

                self.writer.add_scalar('GenPreTraining_train_loss' if what=='train' else 'GenPreTraining_val_loss',loss,self.pretrain_steps)
                
                progress.update(len(images))
                progress.set_postfix(loss=loss.item())        
        
        return gen_loss

    def pretrain_generator(self, epochs):
        print("Pretraining Generator")
        total_loss = 0

        for epoch in range(self.args.pretrain_epochs):

            self.gen.train()
            gen_loss = self.genpretrain_loop('train')
            train_epoch_loss = np.mean(gen_loss)       

            total_loss += train_epoch_loss 

            self.gen.eval()
            gen_loss = self.genpretrain_loop('val')
            val_epoch_loss = np.mean(gen_loss)
            
            if epoch%self.args.pre_log_step == 0:
                print("Epoch {}: \n \t Train: {} \n\t Val: {} ".format(epoch,train_epoch_loss,val_epoch_loss))

            self.pretrain_steps+=1

        return (total_loss/epochs if epochs!=0 else 0)
    
    def adv_loop(self, what):
        total_gen_loss = 0
        total_disc_loss = 0
        
        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=(len(self.train_dataset) if what == 'train' else len(self.dev_dataset))) as progress:
            gen_loss = []
            disc_loss = []
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate((self.adv_train_loader if what=='train' else self.adv_eval_loader)):
        
                images,captions,lengths = images.to(self.args.device), captions.to(self.args.device), lengths.to(self.args.device)
                real_samples = captions #train_data -> (images,lengths,captions)
          
                # self.disc_opt.zero_grad()
                # self.gen_opt.zero_grad()
                if self.cgan:
                    features = self.gen.encoder(images)
#                 fake_samples = self.gen.decoder.sample(features)
         
                gen_samples, _ = self.gen.decoder.sample(features, max_caption_len=max_caption_len)
                fake_samples = gen_samples.detach()

                fake_samples = fake_samples.to(self.args.device)

                real_samples = F.one_hot(real_samples, self.args.vocab_size).float()
#                 fake_samples = F.one_hot(fake_samples, self.args.vocab_size).float()

                # ===Train===
                d_out_real = self.disc(real_samples)
                d_out_fake = self.disc(fake_samples)
                g_out = self.disc(gen_samples)
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
