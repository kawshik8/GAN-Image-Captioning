import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tasks import *


def calc_pp(test_data, gen, epoch, log, args, is_cgan = False):
    gen_loss = []
    gen.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    with torch.no_grad():
        for batch_idx, (images, captions, lengths, max_caption_len) in enumerate(test_data):

            images = images.to(args.device) 
            lengths = lengths.to(args.device)
            attn_mask = captions["attention_mask"].to(args.device)
            captions = captions["input_ids"].to(args.device)

            real_captions = captions             
            
            if is_cgan:
                features = [gen.encoder(images),gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(args.device))]
            else:
                features = gen.decoder.embed(torch.zeros(len(images),1, dtype=torch.long).squeeze(1).to(args.device))
     
            gen_captions, gen_caption_ids = gen.decoder.sample(features, pretrain=True, max_caption_len=max_caption_len)
            real_captions, gen_captions = real_captions.to(args.device), gen_captions.to(args.device)
       
            loss = criterion(gen_captions.view(-1,gen_captions.size(-1)), real_captions.view(-1))
            gen_loss.append(loss.item())

        total_loss = np.mean(gen_loss)
        perplexity = np.exp(total_loss)

    return perplexity
    # log.info('[Epoch %d] Perplexity: %.4f'%(epoch, perplexity))

