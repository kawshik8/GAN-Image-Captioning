import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import get_fixed_temperature, get_losses, create_logger
from torch.utils.tensorboard import SummaryWriter
from generator import *
from new_generator import *
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
        # self.gen = LSTMGenerator(args).to(args.device)
        if args.disc_type == 'cnn':
            self.disc = CDiscriminator(args).to(args.device)
        elif args.disc_type == 'transformer':
            self.disc = TDiscriminator(args).to(args.device)

        self.log = create_logger(__name__, silent=False, to_disk=True,
                                 log_file=args.log_file + ".txt")

        self.sent_log = create_logger(__name__, silent=True, to_disk=True,
                                 log_file=args.sent_log_file + ".txt")

        # Optimizer
        self.pretrain_opt = optim.Adam(self.gen.parameters(), lr=args.pretrain_lr)
        self.gen_opt = optim.Adam(self.gen.parameters(), lr=args.gen_lr, betas=(0.5,0.999))
        self.disc_opt = optim.Adam(self.disc.parameters(), lr=args.disc_lr, betas = (0.5,0.999))

        self.tokenizer = train_dataset.tokenizer

        #Schedulers ReduceLROnPlateau
        # if args.gen_model_type == 'lstm':
        self.pretrain_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.pretrain_opt, patience=args.pretrain_lr_patience, factor=0.5, verbose=True)
        self.gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.gen_opt, patience=args.gen_lr_patience, factor=0.8, verbose=True)
        self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, patience=args.disc_lr_patience, factor=0.8, verbose=True)
        # else:
        #     self.pretrain_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.pretrain_opt, max_lr=5e-4, total_steps = args.pretrain_epochs*((len(train_dataset)//args.pre_train_batch_size)+1), final_div_factor = 10, div_factor=25, pct_start=4/args.pretrain_epochs, anneal_strategy='cos')
        #     self.gen_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.gen_opt, max_lr=5e-4, total_steps = args.adv_epochs*((len(train_dataset)//args.adv_train_batch_size)+1), pct_start=5/args.pretrain_epochs, final_div_factor = 10, div_factor = 25, anneal_strategy='cos')
        #     self.disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.disc_opt, patience=args.disc_lr_patience, verbose=True)
        
        self.pre_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.pre_train_batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
        self.pre_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.pre_eval_batch_size, num_workers=args.num_workers, collate_fn=dev_dataset.collate_fn)

        self.adv_train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.adv_train_batch_size, num_workers=args.num_workers, collate_fn=train_dataset.collate_fn)
        self.adv_eval_loader = DataLoader(dev_dataset, shuffle=False, batch_size=args.adv_eval_batch_size, num_workers=args.num_workers, collate_fn=dev_dataset.collate_fn)
       
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
 
        self.model_dir = args.model_dir
        self.writer = SummaryWriter(args.save_dir)
        
        self.pretrain_steps = 0
        self.gen_train_steps = 0
        self.gen_val_steps = 0
        self.disc_train_steps = 0
        self.disc_val_steps = 0

        self.args = args
        self.cgan = (self.args.conditional_gan==1)
        self.adv_epoch = -1
        self.pretrain_patience = self.args.pretrain_patience
        self.advtrain_patience = self.args.advtrain_patience

        self.num_log_sent = 25
        self.gen_update = 1
        self.teacher_force_choice_pre = 1.0
        self.teacher_force_choice_adv = 0.0

        self.log.info("args: {}".format(self.args))

    def genpretrain_loop(self, what):

        gen_loss = []
        criterion = nn.CrossEntropyLoss(ignore_index=1)
        all_references = []
        all_candidates = []
        num_sent = 0
        total = 0
        dataloader = None
        if what == 'train':
            total = len(self.train_dataset)
            dataloader = self.adv_train_loader
        else:
            total = len(self.dev_dataset)
            dataloader = self.adv_eval_loader

        with open(os.path.join(self.args.save_dir,'./sentences.txt'), 'w') as f:
            with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total = total) as progress:
                for batch_idx, (images, captions, lengths, max_caption_len) in enumerate(dataloader):
                    
                    choice = 0
                    self.pretrain_opt.zero_grad()
                    r_captions = self.train_dataset.convert_to_tokens_references(captions['input_ids'])
                    all_references += r_captions

                    images = images.to(self.args.device) 
                    lengths = lengths.cpu()

                    attn_mask = captions["attention_mask"].to(self.args.device)
                    captions = captions["input_ids"].to(self.args.device)
                    real_captions = captions        
                    
                    if self.cgan:
                        features = self.gen.encoder(images)
                    else:
                        features = None

                    if torch.rand(1) < self.teacher_force_choice_pre:
                        gen_captions, gen_caption_ids = self.gen.decoder(features, captions, lengths, pretrain = True, attn_mask=attn_mask, max_caption_len = max_caption_len)
                    else:
                        gen_captions, gen_caption_ids, _ = self.gen.decoder.sample(features, pretrain = True, max_caption_len = max_caption_len)
                    
                    real_captions, gen_captions = real_captions.to(self.args.device), gen_captions.to(self.args.device)

                    g_captions = self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)
                    all_candidates += g_captions             
        
                    #if not self.args.conditional_gan:
                    #real_captions = real_captions[:,1:]
#                    print("pretrain outputs:")
#                    print(real_captions[:2])
#                    print(gen_caption_ids[:2])
#                    print(real_captions.shape, gen_caption_ids.shape)

                    real_captions = real_captions[:,1:]

                    if what == 'val':
                        f.write('\n\nReal Caption: {}'.format(self.train_dataset.convert_to_tokens_references(captions,skip_special_tokens = False)))
                        f.write('\n\nGenerated Caption: {}'.format(self.train_dataset.convert_to_tokens_candidates(gen_caption_ids,skip_special_tokens = False)))
                        f.flush()

                    loss = criterion(gen_captions.reshape(-1,gen_captions.size(-1)), real_captions.reshape(-1))
                    gen_loss.append(loss.item())

                    if what=='train':
                        loss.backward()
                        self.pretrain_opt.step()
                        #if self.args.gen_model_type == 'transformer':
                        #    self.pretrain_scheduler.step()

                    self.writer.add_scalar('pretrain_losses/Gen_train_loss' if what=='train' else 'pretrain_losses/Gen_val_loss',loss,self.pretrain_steps)         
                    progress.update(len(images))
                    progress.set_postfix(loss=loss.item())#,norm=total_norm)        
                    
        if what == 'val':
            for i in range(10):            
                self.sent_log.info("True Sentence : {} \nPred Sentence : {} \n".format(all_references[i],all_candidates[i]))

        return (gen_loss , all_references, all_candidates)

    def pretrain_generator(self, epochs , weights=[0.25,0.25,0.25,0.25]):
        self.log.info("Pretraining Generator")
        self.sent_log.info("Pretraining Generator")
        total_loss = 0

        best_loss = None
        best_bleu = None
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

            #if self.args.gen_model_type == 'lstm':
            self.pretrain_scheduler.step(val_epoch_loss)

            val_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram -> BLEU-4

            if epoch%self.args.pre_log_step == 0:
                train_perplexity = np.exp(train_epoch_loss)
                val_perplexity = np.exp(val_epoch_loss)
                self.log.info("Epoch {}: \n \t Train Loss: {} \n\t Val Loss: {} \n\t Train PP: {} \n\t Val PP: {} \n\t Train BLEU: {} \n\t Val BLEU: {} " \
                                .format(epoch,train_epoch_loss,val_epoch_loss, train_perplexity, val_perplexity,train_bleu, val_bleu))

            if best_loss is None or val_epoch_loss < best_loss :
                best_loss = val_epoch_loss
                torch.save(self.gen.state_dict(), self.args.model_dir + "/pretrained_model_best_mle.ckpt")
                patience = 0
            elif patience >= self.pretrain_patience:
                self.log.info("Early Stopping at Epoch {}".format(epoch))
                torch.save(self.gen.state_dict(), self.args.model_dir + "/pretrained_model_last.ckpt")
                break
            else:
                patience += 1

            if best_bleu is None or val_bleu > best_bleu:
                best_bleu = val_bleu
                torch.save(self.gen.state_dict(), self.args.model_dir + "/pretrained_model_best_bleu.ckpt")

            if epoch % self.args.checkpoint_freq == 0:
                self.log.info("\n Saved checkpoint at {} ".format(epoch))


            self.writer.add_scalar("pretrain_losses_epoch/val_loss",val_epoch_loss, epoch)
            self.writer.add_scalar("pretrain_losses_epoch/train_loss",train_epoch_loss, epoch)
            self.writer.add_scalar("pretrain_metrics/train_perplexity",train_perplexity,epoch)
            self.writer.add_scalar("pretrain_metrics/val_perplexity",val_perplexity,epoch)
            self.writer.add_scalar("pretrain_metrics/train_bleu",train_bleu,epoch)
            self.writer.add_scalar("pretrain_metrics/val_bleu",val_bleu,epoch)

            self.pretrain_steps+=1

        return (total_loss/epochs if epochs!=0 else 0)
    
    def mle_iteration(self, features, real_captions, lengths, attn_mask, max_caption_len, criterion, what):
        self.gen.eval()
        gen_captions, gen_caption_ids = self.gen.decoder(features, real_captions, lengths, pretrain = True, attn_mask=attn_mask, max_caption_len = max_caption_len)
        gen_captions = gen_captions.to(self.args.device)
#        print("mle outputs:")
#        print(gen_caption_ids[:2])
#        print(real_captions[:2])
#        print(gen_caption_ids.shape, real_captions.shape)
        loss = criterion(gen_captions.reshape(-1,gen_captions.size(-1)), real_captions[:,1:].reshape(-1))
        if what == 'train':
            self.gen.train()
        return loss.detach().cpu()

    def generator_train_iteration(self, generated_captions, what, bce_loss, batch_size, max_caption_len, features, images):
        self.gen_opt.zero_grad()
        total_norm = None
        g_out = self.disc(generated_captions)
        # g_loss = get_losses(g_out=g_out, loss_type=self.args.adv_loss_type)
        if self.args.flip_labels:
            g_loss = bce_loss(g_out, torch.zeros_like(g_out))
        else:
            g_loss = bce_loss(g_out, torch.ones_like(g_out))
        
        if what == 'train':
            g_loss.backward()
            total_norm = 0.0
            for p in self.gen.parameters():
              if p.grad is not None:    
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #print("Gen: ",total_norm)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.args.clip_norm)
            self.gen_opt.step()  

        if self.args.gen_steps > 1 and what == 'train':
            for g_mini_step in range(self.args.gen_steps - 1):

                self.gen_opt.zero_grad()
        
                if self.cgan:
                    features = self.gen.encoder(images)
                gen_captions, gen_caption_ids, output_logits = self.gen.decoder.sample(batch_size, features, states=None, pretrain=False, max_caption_len=max_caption_len)
#                gen_captions, gen_caption_ids, output_logits = self.gen.decoder.sample(batch_size, batch_size, max_caption_len)
                g_out = self.disc(gen_captions)
                g_loss = get_losses(g_out=g_out, loss_type=self.args.adv_loss_type)
                if self.args.flip_labels:
                    g_loss = bce_loss(g_out, torch.zeros_like(g_out))
                else:
                    g_loss = bce_loss(g_out, torch.ones_like(g_out))
                if what == 'train':
                    g_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.args.clip_norm)
                    self.gen_opt.step()

        if what == 'train':
            self.writer.add_scalar('adv_norms/Gen_train_norm',total_norm, self.gen_train_steps)
            self.writer.add_scalar('adv_losses/Generator_train_loss', g_loss.detach().cpu().item(),self.gen_train_steps)
            self.gen_train_steps+=1
        else:
            self.writer.add_scalar('adv_losses/Generator_val_loss', g_loss.detach().cpu().item(),self.gen_val_steps)
            self.gen_val_steps+=1       

        return g_loss.detach().cpu(), total_norm

    def discriminator_train_iteration(self,real_captions, fake_captions,attn_mask, what, bce_loss, batch_size, max_caption_len, features, images):
        total_norm =None
        real_captions = F.one_hot(real_captions, self.args.vocab_size).float()

        self.disc_opt.zero_grad()

        #print(real_captions.shape)
        #if self.args.disc_type == 'cnn':
        d_out_real = self.disc(real_captions) # 16 x 34x 5000
        #elif self.args.disc_type == 'transformer':
        #    d_out_real = self.disc(real_captions.transpose(0,1)).transpose(0,1)
        #print(d_out_real.shape)
        # d_out_fake = self.disc(fake_captions) 
        # d_loss_real, d_loss_fake, d_loss = get_losses(d_out_real, d_out_fake, loss_type=self.args.adv_loss_type)
	
        # d_out_real = self.disc(real_captions)
        real_labels_t = torch.ones(d_out_real.size(0)) - (torch.rand(d_out_real.size(0)) * 0.1)
        fake_labels_t = torch.zeros(d_out_real.size(0)) + (torch.rand(d_out_real.size(0)) * 0.1)

        choice = torch.rand(d_out_real.size(0))
        real_labels = torch.where(choice < 0.05, fake_labels_t, real_labels_t).to(self.args.device)
        choice = torch.rand(d_out_real.size(0))
        fake_labels = torch.where(choice < 0.05, real_labels_t, fake_labels_t).to(self.args.device)

        #print(d_out_real.shape, fake_labels.shape)
        if self.args.flip_labels:
            d_loss_real = bce_loss(d_out_real, fake_labels)
        else:
            d_loss_real = bce_loss(d_out_real, real_labels) #*(torch.rand(d_out_real.size(0))*0.2 + 0.8).to(self.args.device))

        if what == 'train':
            d_loss_real.backward()
            # self.disc_opt.step()

        # self.disc_opt.zero_grad()
       # if self.args.disc_type == 'cnn':
        d_out_fake = self.disc(fake_captions)
       # elif self.args.disc_type == 'transformer':
       #     d_out_real = self.disc(fake_captions.transpose(0,1)).transpose(0,1)
       
        if self.args.flip_labels:
            d_loss_fake = bce_loss(d_out_fake, real_labels)
        else:
            d_loss_fake = bce_loss(d_out_fake, fake_labels)


        if what == 'train':
            d_loss_fake.backward()
            total_norm = 0.0
            for p in self.disc.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            #print("Disc: ",total_norm)
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.args.clip_norm)
            self.disc_opt.step()

        if self.args.disc_steps > 1 and what == 'train':
            for d_mini_step in range(self.args.disc_steps - 1):

                self.disc_opt.zero_grad()     
                d_out_real = self.disc(real_captions)

                if self.args.flip_labels:
                    d_loss_real = bce_loss(d_out_real, fake_labels)
                else:
                    d_loss_real = bce_loss(d_out_real, real_labels)

                d_loss_real.backward()              
                d_out_fake = self.disc(fake_captions)

                if self.args.flip_labels:
                    d_loss_fake = bce_loss(d_out_fake, real_labels)
                else:
                    d_loss_fake = bce_loss(d_out_fake, fake_labels)
                
                d_loss_fake.backward()
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.args.clip_norm)
                self.disc_opt.step()

        d_loss = d_loss_real + d_loss_fake

        if what == 'train':
            self.writer.add_scalar('adv_norms/Disc_train_norm', total_norm, self.disc_train_steps)
            self.writer.add_scalar('adv_losses/Discriminator_train_loss',d_loss.detach().cpu().item(),self.disc_train_steps)
            self.writer.add_scalar('adv_losses/Discriminator_train_fake_loss',d_loss_real.detach().cpu().item(),self.disc_train_steps)
            self.writer.add_scalar('adv_losses/Discriminator_train_real_loss',d_loss_fake.detach().cpu().item(),self.disc_train_steps)
            self.disc_train_steps+=1
        else:
            self.writer.add_scalar('adv_losses/Discriminator_val_loss',d_loss.detach().cpu().item(),self.disc_val_steps)
            self.writer.add_scalar('adv_losses/Discriminator_val_real_loss',d_loss_real.detach().cpu().item(),self.disc_val_steps)
            self.writer.add_scalar('adv_losses/Discriminator_val_fake_loss',d_loss_fake.detach().cpu().item(),self.disc_val_steps)
            self.disc_val_steps+=1

        return d_loss.detach().cpu(), d_loss_real.detach().cpu(), d_loss_fake.detach().cpu(), total_norm

    def adv_loop(self, what):        
        float_epoch = 0.0
        all_references = []
        all_candidates = []
        mle_loss = []
        gen_loss = []
        disc_loss = []
        d_real_loss = []
        d_fake_loss = []
        num_sent = 0
        d_loss = None
        bce_loss = nn.BCEWithLogitsLoss()
        criterion_mle = nn.CrossEntropyLoss(ignore_index=1)
        total = 0
        dataloader = None
        if what == 'train':
            total = len(self.train_dataset)
            dataloader = self.adv_train_loader
        else:
            total = len(self.dev_dataset)
            dataloader = self.adv_eval_loader

        with (torch.enable_grad() if what=='train' else torch.no_grad()), tqdm(total=total) as progress:             
            for batch_idx, (images, captions, lengths, max_caption_len) in enumerate(dataloader):
                
                float_epoch += 1
                print(torch.cuda.memory_allocated(self.args.device))
                all_references += self.train_dataset.convert_to_tokens_references(captions['input_ids'])
                images, lengths = images.to(self.args.device), lengths.to(self.args.device)

                attn_mask = captions["attention_mask"].to(self.args.device)
                real_captions = captions["input_ids"].to(self.args.device)
   
                if self.cgan:
                    features = self.gen.encoder(images)
                else:
                    features = None
   
                gen_captions, gen_caption_ids, output_logits = self.gen.decoder.sample(images.size(0), features, states=None,pretrain=False, max_caption_len=max_caption_len)
                # gen_captions, gen_caption_ids, output_logits = self.gen.sample(images.size(0), images.size(0), max_caption_len)
                fake_captions = gen_captions.detach()   
                
               # if not self.cgan:              
               # real_captions = real_captions[:,1:]   # Remove start token   
               #     fake_captions = fake_captions[:,:-1]   #Remove token generated for stop token (we generate sentence upto max length in sampling)
               #     gen_captions = gen_captions[:,:-1]
               #     output_logits = output_logits[:, :-1]

                real_captions =  real_captions.to(self.args.device)
#                print("adv outputs:")
#                print(gen_caption_ids[:2])
#                print(real_captions[:2])
#                print(gen_caption_ids.shape, real_captions.shape)
                loss = self.mle_iteration(features, real_captions, lengths, attn_mask, max_caption_len, criterion_mle, what)
                mle_loss.append(loss.detach().item())
                if what == 'train':               
                    self.writer.add_scalar('adv_losses/gen_train_mle_loss',loss.item(),self.gen_train_steps)
                else:
                    self.writer.add_scalar('adv_losses/gen_val_mle_loss',loss.item(),self.gen_val_steps)

                all_candidates+=self.train_dataset.convert_to_tokens_candidates(gen_caption_ids)


                # ===Train===

                #Discriminator
                d_loss, d_real, d_fake, total_norm_d= self.discriminator_train_iteration(real_captions, fake_captions, attn_mask, what, bce_loss,images.size(0), max_caption_len, features, images)
                d_fake_loss.append(d_fake.item())
                d_real_loss.append(d_real.item())
                disc_loss.append(d_loss.item())


                #Generator
                g_loss, total_norm_g = self.generator_train_iteration(gen_captions, what, bce_loss,images.size(0), max_caption_len, features, images)
                gen_loss.append(g_loss.item())

#                self.writer.add_scalars('' if what == 'train' else 'adv_losses/gen_disc_val_loss', {
#                                    '': d_loss.item(),
#                                    '':g_loss.item()
#                                }, self.gen_steps)
#                self.writer.add_scalars('' if what == 'train' else 'adv_losses/gen_discRF_val_loss', {
#                                    '':d_real.item(),
#                                    '':d_fake.item(),
#                                    '':g_loss.item()
#                                }, self.gen_steps)
#                self.writer.add_scalars('', {
#                                    'adv_losses/discRF_real':d_real.item(),
#                                    'adv_losses/discRF_fake':d_fake.item(),
#                                    'adv_losses/discRF_total':d_loss.item()
#                                }, self.gen_steps)



                # g_loss = self.generator_train_iteration(gen_captions, what, bce_loss)
                # gen_loss.append(g_loss.item())

                progress.update(len(images))
                progress.set_postfix(disc_loss=d_loss.item(), gen_loss=g_loss.item(), mle_loss=loss.item(),d_norm = total_norm_d, g_norm = total_norm_g)

                if what == 'train':
                    self.update_temperature(self.adv_epoch + (float_epoch/len(dataloader)), self.args.adv_epochs)  # update temperature
                    self.writer.add_scalar('temperature',self.gen.decoder.temperature,self.gen_train_steps)

        total_gen_loss = np.mean(gen_loss)
        total_disc_loss = np.mean(disc_loss)
        total_mle_loss = np.mean(mle_loss)
        total_d_real = np.mean(d_real_loss)
        total_d_fake = np.mean(d_fake_loss)

        for i in range(10):
            self.sent_log.info("True Sentence : {} \nPred Sentence : {} \n".format(all_references[i],all_candidates[i]))

        return (total_gen_loss, total_disc_loss, total_mle_loss, all_references, all_candidates, total_d_real, total_d_fake)

    def update_temperature(self, i, N):
        self.gen.decoder.temperature = get_fixed_temperature(self.args.temperature, i, N, self.args.temp_adpt)

    def _run(self, weights=[0.25,0.25,0.25,0.25]):
    
        ## === PRETRAINING GENERATOR === ##
        if self.args.pretrained_model_file is not None:
            self.log.info('Loading pretrained model file')
            self.gen.load_state_dict(torch.load(self.args.pretrained_model_file))
        else:
            self.pretrain_generator(self.args.pretrain_epochs, weights)

        # # ===ADVERSARIAL TRAINING===
        self.log.info('Starting Adversarial Training...')
        self.sent_log.info('Starting Adversarial Training...')
        
        patience = 0
        best_loss = None
        best_mle_loss = None
        best_bleu = None
        
        for adv_epoch in range(self.args.adv_epochs):

            self.adv_epoch = adv_epoch
            self.sent_log.info("\nEpoch : {}".format(adv_epoch))
            
            self.disc.train()
            self.gen.train()
            train_g_loss, train_d_loss, train_mle_loss, ref, gen, train_d_real, train_d_fake = self.adv_loop('train')  # Discriminator
            train_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram BLEU-4
            
            self.disc.eval()
            self.gen.eval()
            val_g_loss, val_d_loss, val_mle_loss, ref, gen, val_d_real, val_d_fake = self.adv_loop('val')
            val_bleu = nltk.translate.bleu_score.corpus_bleu(ref,gen,weights = weights,smoothing_function=SmoothingFunction().method1) #Default 4 gram BLEU-4
            # TEST
            if adv_epoch % self.args.adv_log_step == 0 or adv_epoch == self.args.adv_epochs - 1:
                val_perplexity = np.exp(train_mle_loss)
                train_perplexity = np.exp(val_mle_loss)
                self.log.info('[ADV] epoch %d (temperature: %.4f):\n\t g_loss: %.4f | %.4f \n\t d_loss: %.4f | %.4f \n\t Total_d_real: %.4f | %.4f \n\t Total_d_fake: %.4f | %.4f \n\t Train PP: %.4f \n\t Val PP: %.4f \n\t Train BLEU: %.4f \n\t Val BLEU: %.4f' %\
                              (adv_epoch, self.gen.decoder.temperature, train_g_loss, val_g_loss, train_d_loss, val_d_loss,train_d_real, val_d_real,train_d_fake, val_d_fake, train_perplexity, val_perplexity, train_bleu, val_bleu))
            
            self.writer.add_scalar('adv_losses_epoch/gen_train_loss',train_g_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_train_loss',train_d_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/train_mle_loss',train_mle_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_train_real_loss',train_d_real,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_train_fake_loss',train_d_fake,adv_epoch)
            self.writer.add_scalar('adv_metrics/train_bleu',train_bleu,adv_epoch)
            self.writer.add_scalar('adv_metrics/train_perplexity',np.exp(train_mle_loss),adv_epoch)
            
            self.writer.add_scalar('adv_losses_epoch/gen_val_loss',val_g_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_val_loss',val_d_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/val_mle_loss',val_mle_loss,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_val_real_loss',val_d_real,adv_epoch)
            self.writer.add_scalar('adv_losses_epoch/disc_val_fake_loss',val_d_fake,adv_epoch)
            self.writer.add_scalar('adv_metrics/val_bleu',val_bleu,adv_epoch)
            self.writer.add_scalar('adv_metrics/val_perplexity',np.exp(val_mle_loss),adv_epoch)

            if self.args.gen_model_type == 'lstm':
                self.gen_scheduler.step(val_g_loss)
            self.disc_scheduler.step(val_d_loss)

            if best_loss is None or val_g_loss < best_loss :
                best_loss = val_g_loss 
                self.log.info("\n Best g_loss found ! {} ".format(best_loss))
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model_best_gloss.ckpt")
                patience = 0
            # elif patience >= self.advtrain_patience:
            #     self.log.info("Early Stopping at Epoch {}".format(adv_epoch))
            #     break
            else:
                patience += 1


            if best_mle_loss is None or val_mle_loss < best_mle_loss:
                best_mle_loss = val_mle_loss
                self.log.info("\n Best mle_loss found ! {} ".format(best_mle_loss))
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model_best_mle.ckpt")

            
            if adv_epoch % self.args.checkpoint_freq == 0:
                self.log.info("\n Saved checkpoint at {} ".format(adv_epoch))
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model_checkpoint.ckpt")

            if best_bleu is None or val_bleu > best_bleu:
                best_bleu = val_bleu
                self.log.info("\n Best bleu found ! {} ".format(best_bleu))
                torch.save({"generator":self.gen.state_dict(),
                            "discriminator":self.disc.state_dict()}, self.model_dir + "/adv_model_best_bleu.ckpt")
