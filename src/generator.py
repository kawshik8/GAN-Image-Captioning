import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, args):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, args.gen_embed_size)
        self.bn = nn.BatchNorm1d(args.gen_embed_size, momentum=0.01)
        self.args = args

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class Decoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.embed_dim)
        self.lstm = nn.LSTM(args.embed_dim, args.gen_hidden_dim, args.num_layers, batch_first=True)
        self.linear = nn.Linear(args.hidden_dim, args.vocab_dim)
        self.max_seg_length = args.max_seq_length
        self.temperature = args.temperature
        self.args = args
        
    def forward(self, features, caps, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(caps)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  #First timestep input to lstm is features from image. Appending to captions
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        packed_output, hidden = self.lstm(packed)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True) #Unpack sequence

        gumbel_t = self.add_gumbel(self.linear(output))
        pred = F.softmax(gumbel_t * self.temperature, dim=-1) 

        return pred, hidden
    
    def sample(self, features, states=None):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids
    
    @staticmethod
    def add_gumbel(o_t, eps=1e-10, gpu=0):
        """Add o_t by a vector sampled from Gumbel(0,1)"""
        u = torch.zeros(o_t.size())

        u = u.to(self.args.device)
            
        u.uniform_(0, 1)
        g_t = -torch.log(-torch.log(u + eps) + eps)

        g_t = g_t.to(self.args.device)

        gumbel_t = o_t + g_t
        return gumbel_t

class Generator(nn.Module):
    def __init__(self, args):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Generator, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.args = args
        self.init_params()
        
    def forward(self, images, caps, lengths):
        """Extract feature vectors from input images."""
        features = self.encoder(images)
        pred, hidden = self.decoder(features, caps, lengths)
        return pred, hidden

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.args.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.args.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
