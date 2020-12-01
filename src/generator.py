import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer import *
from torch.autograd import Variable
import numpy as np
from utils import init_weight

def get_resnet_model(model_type):
    if model_type == 'resnet50':
        return models.resnet50(pretrained=True,progress=True)
    elif model_type == 'resnet34':
        return models.resnet34(pretrained=True,progress=True)
    elif model_type == 'resnet18':
        return models.resnet18(pretrained=True,progress=True)
    elif model_type == 'resnet101':
        return models.resnet101(pretrained=True,progress=True)
    elif model_type == 'resnet152':
        return models.resnet152(pretrained=True,progress=True)

OUT_IMAGE_FEATURE_DIMENSION_DICT = {"resnet18": 512, "resnet34":512, "resnet50":2048, "resnet101":2048, "resnet152":2048}

class Encoder(nn.Module):
    def __init__(self, args):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Encoder, self).__init__()
        resnet = get_resnet_model(args.resnet_type)
        modules = list(resnet.children())[:-2]      # delete the last fc layer and avg pool 2d.
        self.resnet = nn.Sequential(*modules) 

        if args.gen_model_output == 'transformer':
            self.conv = nn.Conv2d(OUT_IMAGE_FEATURE_DIMENSION_DICT[args.resnet_type], args.gen_embed_dim, 3, 1)
            self.bn = nn.BatchNorm2d(args.gen_embed_dim, momentum=0.01)
        else:
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            self.linear = nn.Linear(resnet.fc.in_features, args.gen_embed_dim)
            self.bn = nn.BatchNorm1d(args.gen_embed_dim, momentum=0.01)

        self.args = args
        self.freeze_cnn = args.freeze_cnn

    def forward(self, images):
        """Extract feature vectors from input images."""
        with (torch.no_grad() if self.freeze_cnn else torch.enable_grad()):
            features = self.resnet(images)

        if self.args.gen_model_output == 'transformer':
            features = self.bn(self.conv(features))
        else:
            features = self.pool(features)
            features = features.reshape(features.size(0), -1)
            features = self.bn(self.linear(features))

        return features

class Decoder(nn.Module):
    def __init__(self, args):
        """Set the hyper-parameters and build the layers."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(args.vocab_size, args.gen_embed_dim, padding_idx=1)

        if args.gen_model_type == 'lstm':
            self.lstm = nn.LSTM(args.gen_embed_dim, args.gen_hidden_dim, args.gen_num_layers, batch_first=True, bidirectional=True)


        elif args.gen_model_type == 'transformer':# and args.conditional_gan:
            if args.conditional_gan:
                decoder_layer = TransformerDecoderLayer(args.gen_embed_dim, args.gen_nheads, args.gen_hidden_dim)
                self.transformer = TransformerDecoder(decoder_layer, args.gen_num_layers)


            else:#if d not args.conditional_gan:
                encoder_layer = TransformerEncoderLayer(args.gen_embed_dim, args.gen_nheads, args.gen_hidden_dim)
                self.transformer = TransformerEncoder(encoder_layer, args.gen_num_layers)
            self.pos_embed = nn.Embedding(100, args.gen_embed_dim)


        if args.gen_model_type == 'lstm':
            self.linear = nn.Linear(args.gen_hidden_dim * 2, args.vocab_size)
        else:
            self.linear = nn.Linear(args.gen_embed_dim, args.vocab_size)
        self.max_seq_length = args.max_seq_len
        self.temperature = args.temperature
        self.args = args
        #self.device = args.device
        
    def forward(self, image_features, caps, lengths, pretrain=False, attn_mask = None, max_caption_len=34):
        """Decode image feature vectors and generates captions."""
        # print(lengths)  
        embeddings = self.embed(caps) #[0,.....,2,1,1,1]

        if self.args.conditional_gan:
            if self.args.gen_model_type == 'transformer':
                inputs = embeddings
                image_features = image_features.view(image_features.size(0),-1,image_features.size(1))
            else:
                inputs = torch.cat((image_features.unsqueeze(1), embeddings), 1) 
        else:
             inputs = embeddings
             lengths-=1

        if self.args.gen_model_type == 'lstm':
            # output, hidden = self.lstm(inputs) 
            # output = output[:,:-1]
            packed = pack_padded_sequence(inputs, lengths, batch_first=True)   
            packed_output, hidden = self.lstm(packed, )          
            output, input_sizes = pad_packed_sequence(packed_output, batch_first=True) #Unpack sequence
           
        else:
            attn_mask = (1.0 - attn_mask).type(torch.bool)
            
            no_peek_mask = np.triu(np.ones((max_caption_len, max_caption_len)), k=1)  
            no_peek_mask = Variable(torch.from_numpy(no_peek_mask) == 1).type(torch.bool).to(self.args.device)
            #TODO 
            # attn_mask = attn_mask & no_peek_mask
            # print(inputs.shape)
            positions = self.pos_embed(torch.arange(inputs.size(1)).long().unsqueeze(0).repeat(inputs.size(0),1).to(self.args.device)).transpose(0,1)

            if self.args.conditional_gan:
                output = self.transformer(inputs.transpose(0,1), memory = image_features, query_pos=positions, tgt_mask = no_peek_mask, tgt_key_padding_mask=attn_mask).transpose(0,1)
            else:
                output = self.transformer(inputs.transpose(0,1), pos=positions, mask = no_peek_mask, src_key_padding_mask=attn_mask).transpose(0,1)
                output =output[:,:-1]

        if pretrain:
            pred = self.linear(output)
        else:
            gumbel_t = self.add_gumbel(self.linear(output))
            pred = F.softmax(gumbel_t * self.temperature, dim=-1)

        pred_index = pred.max(-1)[1]
        return pred, pred_index
    
    def sample(self, batch_size, image_features, states=None, pretrain=False,max_caption_len=34):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []
        
        if self.args.conditional_gan:  
            if self.args.gen_model_type == 'transformer':            
                if args.gen_model_output == 'pool':
                    image_features = image_features.unsqueeze(1) 
                else:
                    image_features = image_features.view(image_features.size(0),-1,image_features.size(1))
                inputs = self.embed(torch.zeros(batch_size,1, dtype=torch.long).to(self.args.device))
            else:
                inputs = image_features.unsqueeze(1)
        else:
            inputs = self.embed(torch.zeros(batch_size,1, dtype=torch.long).to(self.args.device))

        outputs = []
        for i in range(max_caption_len):

            if self.args.gen_model_type == 'lstm':       
                hiddens, states = self.lstm(inputs, states)                    
            else:
                positions = self.pos_embed(torch.arange(inputs.size(1)).long().unsqueeze(0).repeat(inputs.size(0),1).to(self.args.device)).transpose(0,1)
                
                if self.args.conditional_gan:
                    hiddens = self.transformer(tgt=inputs.transpose(0,1),query_pos = pos, memory=image_features.transpose(0,1)).transpose(0,1)
                else:         
                    hiddens = self.transformer(inputs.transpose(0,1), pos = positions).transpose(0,1)

            if pretrain:
                pred = self.linear(hiddens)
                outputs.append(pred[:,-1])
   
            else:
                gumbel_t = self.add_gumbel(self.linear(hiddens))            # outputs:  (batch_size, vocab_size)
                pred = F.softmax(gumbel_t * self.temperature, dim=-1)  
                outputs.append(pred[:,-1])
            
            pred_index = pred.max(-1)[1]
            pred_index = pred_index[:,-1]                                     # predicted: (batch_size)          
            sampled_ids.append(pred_index)

            if self.args.gen_model_type == 'lstm':
                inputs = self.embed(pred_index.detach()).unsqueeze(1)        # inputs: (batch_size,1, embed_size)
              
            elif self.args.gen_model_type == 'transformer':
                inputs = torch.cat([inputs,self.embed(pred_index.detach().unsqueeze(1))],1)                    # inputs: (batch_size, n+1, embed_size)

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        outputs = torch.stack(outputs, 1)
        return outputs, sampled_ids
    
    def add_gumbel(self, o_t, eps=1e-10, gpu=0):
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
        """Load the pretrained ResNet and replace top fc layer."""
        super(Generator, self).__init__()

        if args.conditional_gan:
            self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        self.args = args
        init_weight(self)
        
    def forward(self, images, caps, lengths, pretrain=False):
        """Extract feature vectors from input images."""
        if self.args.cgan:
            features = self.encoder(images)
        else:
            features = None
        pred, hidden = self.decoder(features, caps, lengths, pretrain)
        return pred, hidden

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if self.args.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif self.args.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)

if __name__=='__main__':
    from args import get_args

    args = get_args()
    generator = Generator(args)

    b = 16
    sample_images = torch.rand(b,3,256,256, dtype=torch.float).to(args.device)
    sample_captions = torch.ones(b,34, dtype=torch.long).to(args.device)
    sample_lengths = (torch.ones(b, dtype=torch.long)*34).to(args.device)

    if args.conditional_gan:
        features = generator.encoder(sample_images)
        print("image features: ",generator.encoder(sample_images).shape)
    else:
        features = None
        

    pred, pred_index = generator.decoder(features, sample_captions, sample_lengths, attn_mask=None, max_caption_len = 34)
    # pretrain_pred,_ = generator.decoder.sample(features, pretrain=True)

    # adv_pred,_ = generator.decoder.sample(features)

    print(pred.shape, pred_index.shape)

