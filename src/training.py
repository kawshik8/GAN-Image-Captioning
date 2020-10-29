import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import config as cfg
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import get_fixed_temperature, get_losses, create_logger
from torch.utils.tensorboard import SummaryWriter
from generator import *
from discriminator import *
import numpy as np

class RelGANInstructor():
    def __init__(self, train_loader, dev_loader):

        # generator, discriminator
        self.gen = Generator().to(cfg.device)
        self.disc = Discriminator().to(cfg.device)
        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=cfg.log_filename if cfg.if_test
                                 else [cfg.log_filename, cfg.save_root + 'log.txt'])
        # Optimizer
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_lr)
        self.gen_adv_opt = optim.Adam(self.gen.parameters(), lr=cfg.gen_adv_lr)
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=cfg.dis_lr)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        
        self.writer = SummaryWriter()
        
        self.pretrain_steps = 0
        self.gen_steps = 0
        self.disc_steps = 0

    # def __del__(self):
    #     self.writer.close()

    def _run(self):

        ## === PRETRAINING GENERATOR === ##
        self.pretrain_generator(cfg.PRETRAIN_EPOCHS)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        progress = tqdm(range(cfg.ADV_train_epoch))

        for adv_epoch in progress:
            train_d_loss = self.adv_train_discriminator(cfg.ADV_d_step, 'train')  # Discriminator
            val_d_loss = self.adv_train_discriminator(1, 'val')

            train_g_loss = self.adv_train_generator(cfg.ADV_g_step,'train')  # Generator
            val_g_loss = self.adv_train_generator(1,'val')

            self.update_temperature(adv_epoch, cfg.ADV_train_epoch)  # update temperature
    
            progress.set_description(
                'g_loss: %.4f, d_loss: %.4f, temperature: %.4f' % (g_loss, d_loss, self.gen.decoder.temperature))

            # TEST
            if adv_epoch % cfg.adv_log_step == 0 or adv_epoch == cfg.ADV_train_epoch - 1:
                self.log.info('[ADV] epoch %d: g_loss: %.4f, d_loss: %.4f' % (
                    adv_epoch, g_loss, d_loss))

                # if cfg.if_save and not cfg.if_test:
                #     self._save('ADV', adv_epoch)

        progress.close()

    def genpretrain_loop(self, what):

        gen_loss = []
        with (torch.grad() if what=='train' else torch.nograd()):
            for batch_idx, (images, captions, lengths) in enumerate((self.train_loader if what=='train' else self.dev_loader)):
                real_samples = captions 
                self.gen_opt.zero_grad()
                
                gen_samples,_ = self.gen(images, captions, lengths)
                if cfg.cuda:
                    real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

                targets = pack_padded_sequence(real_samples, lengths, batch_first=True, enforce_sorted=False)[0]
                outputs = pack_padded_sequence(gen_samples, lengths, batch_first=True, enforce_sorted=False)[0]      

                criterion = nn.CrossEntropyLoss()
                loss = torch.autograd.Variable(criterion(outputs, targets), requires_grad=True)
                if what == 'train':
                    self.optimize(self.gen_opt, loss, self.gen)

                gen_loss.append(loss.item())

                self.writer.add_scalar(f'GenPreTraining_{what}_loss',loss,self.pretrain_steps)
                
    
        return gen_loss

    def pretrain_generator(self, epochs):
        print("Pretraining Generator")
        total_loss = 0

        progress = tqdm(range(cfg.ADV_train_epoch))
        for epoch in progress:

            gen_loss = self.genpretrain_loop('train')
            train_epoch_loss = np.mean(gen_loss)       

            total_loss += train_epoch_loss 

            gen_loss = self.genpretrain_loop('val')
            val_epoch_loss = np.mean(gen_loss)

            progress.set_description(
                    'pretrain_val_gen_loss: %.4f, pretrain_val_gen_loss' % (train_epoch_loss, val_epoch_loss))
            
            if epoch%cfg.pre_log_step == 0:
                print("Epoch {}: \n \t Train: {} \n\t Val: {} ".format(epoch,train_epoch_loss,val_epoch_loss))

            self.pretrain_steps+=1

        return total_loss/epochs
    
    def adv_train_generator(self, g_step, what):
        total_loss = 0
        
        with (torch.grad() if what=='train' else torch.nograd()):
            for step in range(g_step):
                gen_loss = []
                for batch_idx, (images, captions, lengths) in enumerate((self.train_loader if what=='train' else self.dev_loader)):
                    real_samples = captions #train_data -> (images,lengths,captions)
                    features = self.gen.encoder(images)
                    gen_samples = self.gen.decoder.sample(features)

                    if cfg.cuda:
                        real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

                    real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
                    gen_samples = F.one_hot(gen_samples, cfg.vocab_size).float()

                    # ===Train===
                    d_out_real = self.disc(real_samples)
                    d_out_fake = self.disc(gen_samples)
                    g_loss, _ = get_losses(d_out_real, d_out_fake, cfg.loss_type)

                    if what == 'train':
                        self.optimize(self.gen_adv_opt, g_loss, self.gen)

                    self.writer.add_scalar(f'Generator_{what}_loss',g_loss,self.gen_steps)
                    self.gen_steps+=1
                    gen_loss.append(g_loss.item())

                total_loss += np.mean(gen_loss)

        return total_loss / g_step if g_step != 0 else 0

    def adv_train_discriminator(self, d_step, what):
        total_loss = 0
        with (torch.grad() if what=='train' else torch.nograd()):
            for step in range(d_step):
                dis_loss = []
                for batch_idx, (images, captions, lengths) in enumerate((self.train_loader if what=='train' else self.dev_loader)):
                    real_samples = captions

                    features = self.gen.encoder(images)
                    gen_samples = self.gen.decoder.sample(features).detach()

                    if cfg.cuda:
                        real_samples, gen_samples = real_samples.cuda(), gen_samples.cuda()

                    real_samples = F.one_hot(real_samples, cfg.vocab_size).float()
                    gen_samples = F.one_hot(gen_samples, cfg.vocab_size).float()

                    # ===Train===
                    d_out_real = self.disc(real_samples)
                    d_out_fake = self.disc(gen_samples)
                    _, d_loss = get_losses(d_out_real, d_out_fake, cfg.loss_type)

                    if what == 'train':
                        self.optimize(self.disc_opt, d_loss, self.disc)

                    self.writer.add_scalar(f'Discriminator_{what}_loss',d_loss,self.disc_steps)
                    self.disc_steps+=1
                    dis_loss.append(d_loss.item())

                total_loss += np.mean(dis_loss)

        return total_loss / d_step if d_step != 0 else 0

    def update_temperature(self, i, N):
        self.gen.decoder.temperature = get_fixed_temperature(cfg.temperature, i, N, cfg.temp_adpt)

    @staticmethod
    def optimize(opt, loss, model=None, retain_graph=False):
        opt.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()