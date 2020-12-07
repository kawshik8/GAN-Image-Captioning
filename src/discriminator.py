import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import init_weight
from transformer import *

class CDiscriminator(nn.Module):
    def __init__(self, args, gpu=False,dropout=0.2):
        super(CDiscriminator, self).__init__()

        self.vocab_size = args.vocab_size
        self.embed_dim = args.disc_embed_dim
        self.padding_idx = args.padding_idx
        self.feature_dim = sum(args.disc_num_filters)
        self.emb_dim_single = int(args.disc_embed_dim / args.disc_num_rep)
        self.gpu = gpu

        self.embeddings = nn.Linear(self.vocab_size, self.embed_dim, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, self.emb_dim_single), stride=(1, self.emb_dim_single)) for (n, f) in
            zip(args.disc_num_filters, args.disc_filter_sizes)
        ])

        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)
        self.args = args
        # self.apply(init_weight)
        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        cons = [F.leaky_relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.leaky_relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.leaky_relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.leaky_relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.args.disc_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.args.disc_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)

#if __name__=='__main__':
#    from args import get_args
#
#    args = get_args()
#    discriminator = Discriminator(args)

 ##   b = 16
#    real_captions = F.one_hot(torch.ones(b,34,dtype=torch.long),args.vocab_size).float().to(args.device)
#    fake_captions = torch.rand(b,34,args.vocab_size).to(args.device)
#    print(real_captions.shape, fake_captions.shape)
#
#    real_pred = discriminator(real_captions#)
#    fake_pred = discriminator(fake_captions)#
#    print("output fake and real : ",fake_pred.shape, real_pred.shape)


class TDiscriminator(nn.Module):
     def __init__(self, args, gpu=False,dropout=0.2):
         super(TDiscriminator, self).__init__()

         self.args = args
         self.embeddings = nn.Linear(args.vocab_size, args.disc_embed_dim)
         self.pos_embeddings = nn.Embedding(100, args.disc_embed_dim)

         encoder_layer = TransformerEncoderLayer(args.disc_embed_dim, args.disc_nheads, args.disc_hidden_dim)
         self.transformer = TransformerEncoder(encoder_layer, args.disc_num_layers)

         self.out2logits = nn.Linear(args.disc_embed_dim, 1)

         self.init_params()


     def forward(self, input):
        
         embeddings = self.embeddings(input) # 16 x 34 x 50234 -> 16 x34 x 32

         positions = self.pos_embeddings(torch.arange(embeddings.size(1)).unsqueeze(0).repeat(embeddings.size(0),1).to(embeddings.device)).transpose(0,1)

         # print(embeddings.shape, positions.shape)

         out = self.transformer(embeddings.transpose(0,1), pos = positions).transpose(0,1)[:,0]
#         print(out.shape)

         logits = self.out2logits(out).squeeze()
#         print(logits.shape)
         return logits

     def init_params(self):
         for param in self.parameters():
             if param.requires_grad and len(param.shape) > 0:
                 stddev = 1 / math.sqrt(param.shape[0])
                 if self.args.disc_init == 'uniform':
                     torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                 elif self.args.disc_init == 'normal':
                     torch.nn.init.normal_(param, std=stddev)

if __name__=='__main__':
    from args import get_args

    args = get_args()
    discriminator = TDisc(args)

    b = 16
    real_captions = F.one_hot(torch.ones(b,34,dtype=torch.long),args.vocab_size).float().to(args.device)
    fake_captions = torch.rand(b,34,args.vocab_size).to(args.device)
    print(real_captions.shape, fake_captions.shape)

    real_pred = discriminator(real_captions)
    fake_pred = discriminator(fake_captions)
    print("output fake and real : ",fake_pred.shape, real_pred.shape)
