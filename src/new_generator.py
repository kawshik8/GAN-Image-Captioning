import math
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import torch.nn.functional as F

class LSTMGenerator(nn.Module):
    def __init__(self, args, padding_idx = 1):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'

        self.hidden_dim = args.gen_hidden_dim
        self.embedding_dim = args.gen_embed_dim
        self.num_layers = args.gen_num_layers
        self.vocab_size = args.vocab_size
        self.padding_idx = padding_idx
        self.device = args.device

        self.temperature = 1.0

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,self.num_layers, batch_first=True)
        self.lstm2out = nn.Linear(self.hidden_dim, self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.args = args
        self.init_params()

    def step(self, inp, hidden):
        """
        RelGAN step forward
        :param inp: [batch_size]
        :param hidden: memory size
        :return: pred, hidden, next_token, next_token_onehot, next_o
            - pred: batch_size * vocab_size, use for adversarial training backward
            - hidden: next hidden
            - next_token: [batch_size], next sentence token
            - next_token_onehot: batch_size * vocab_size, not used yet
            - next_o: batch_size * vocab_size, not used yet
        """
        emb = self.embeddings(inp).unsqueeze(1)
        out, hidden = self.lstm(emb, hidden)
        output_logits = self.lstm2out(out.squeeze(1))

        gumbel_t = self.add_gumbel(output_logits)
        next_token = torch.argmax(gumbel_t, dim=1).detach()
        # next_token_onehot = F.one_hot(next_token, cfg.vocab_size).float()  # not used yet
        next_token_onehot = None

        pred = F.softmax(gumbel_t * self.temperature, dim=-1)  # batch_size * vocab_size
        # next_o = torch.sum(next_token_onehot * pred, dim=1)  # not used yet
        next_o = None

        return pred, hidden, next_token, next_token_onehot, next_o, output_logits

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size,max_seq_len, one_hot=False, start_letter=0):
        """
        Sample from RelGAN Generator
        - one_hot: if return pred of RelGAN, used for adversarial training
        :return:
            - all_preds: batch_size * seq_len * vocab_size, only use for a batch
            - samples: all samples
        """
        global all_preds
        outputs = torch.zeros(batch_size, max_seq_len, self.vocab_size).to(self.device)
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, max_seq_len).long()
        # if one_hot:
        all_preds = torch.zeros(batch_size, max_seq_len, self.vocab_size)
        # if self.gpu:
        all_preds = all_preds.to(self.device)

        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            # if self.gpu:
            inp = inp.to(self.device)

            for i in range(max_seq_len):
                pred, hidden, next_token, _, _, output_logits = self.step(inp, hidden)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token
                # if one_hot:
                all_preds[:, i] = pred
                inp = next_token
                outputs[:,i] = output_logits

        samples = samples[:num_samples]  # num_samples * seq_len

        # if one_hot:
            # return all_preds  # batch_size * seq_len * vocab_size
        return all_preds, samples, outputs

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.args.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.args.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)

    def init_hidden(self, batch_size=16):
        h = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)

        # if self.gpu:
        #     return h.cuda(), c.cuda()
        # else:
        #     return h, c

    def add_gumbel(self,o_t, eps=1e-10):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        # print(o_t, type(o_t))
        u = torch.zeros(o_t.size()).to(self.device)
        # if gpu:
        #     u = u.cuda()

        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)
        gumbel_t = o_t + g_t
        return gumbel_t

if __name__=='__main__':
    from args import get_args

    args = get_args()
    generator = LSTMGenerator(args)

    b = 4
    max_len = 10
    # sample_images = torch.rand(b,3,256,256, dtype=torch.float).to(args.device)
    # sample_captions = torch.ones(b,max_len, dtype=torch.long).to(args.device)
    gen_samples = generator.sample(b, b, 3, one_hot=False)
    print(gen_samples)
    print(gen_samples.shape)