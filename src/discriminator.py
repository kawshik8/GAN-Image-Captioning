import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Discriminator(nn.Module):
    def __init__(self, args, gpu=False,dropout=0.2):
        super(Discriminator, self).__init__()

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
        self.init_params()

    def forward(self, inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        # print(emb.shape)
        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        # for con in cons:
        #     print(con.shape)
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        # for pool in pools:
        #     print(pool.shape)
        # print(pools.shape)
        pred = torch.cat(pools, 1)
        # print(pred.shape)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        # print(pred.shape)
        highway = self.highway(pred)
        # print(highway.shape)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway
        # print(pred.shape)

        pred = self.feature2out(self.dropout(pred))
        # print(pred.shape)
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

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
    discriminator = Discriminator(args)

    b = 16
    real_captions = F.one_hot(torch.ones(b,34,dtype=torch.long),args.vocab_size).float().to(args.device)
    fake_captions = torch.rand(b,34,args.vocab_size).to(args.device)
    print(real_captions.shape, fake_captions.shape)

    real_pred = discriminator(real_captions)
    fake_pred = discriminator(fake_captions)
    print("output fake and real : ",fake_pred.shape, real_pred.shape)


