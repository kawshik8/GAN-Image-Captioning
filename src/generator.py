import torch
import math
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformer import *

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

        if args.gen_model_output == 'grid' and args.gen_model_type != 'lstm':
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

        if args.gen_model_output == 'grid' and args.gen_model_type != 'lstm':
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
        self.embed = nn.Embedding(args.vocab_size, args.gen_embed_dim)
        if args.gen_model_type == 'lstm':
            self.lstm = nn.LSTM(args.gen_embed_dim, args.gen_hidden_dim, args.gen_num_layers, batch_first=True)
        elif args.gen_model_type == 'transformer' and args.conditional_gan:
            decoder_layer = TransformerDecoderLayer(args.gen_hidden_dim)
            self.transformer = TransformerDecoder(decoder_layer, 6)
        elif args.gen_model_type == 'transformer' and not args.conditional_gan:
            encoder_layer = TransformerEncoderLayer(args.gen_hidden_dim)
            self.transformer = TransformerEncoder(encoder_layer, 6)
        self.linear = nn.Linear(args.gen_hidden_dim, args.vocab_size)
        self.max_seq_length = args.max_seq_len
        self.temperature = args.temperature
        self.args = args
        #self.device = args.device
        
    def forward(self, features, caps, lengths, pretrain=False):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(caps)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)  #First timestep input to lstm is features from image. Appending to captions
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False) 
        packed_output, hidden = self.lstm(packed)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True) #Unpack sequence

        if pretrain:
            pred = self.linear(output)
        else:
            gumbel_t = self.add_gumbel(self.linear(output))
            pred = F.softmax(gumbel_t * self.temperature, dim=-1) 

        return pred, hidden
    
    def sample(self, features, states=None, pretrain=False, max_caption_len=34):
        """Generate captions for given image features using greedy search."""
        sampled_ids = []

        # print(features.shape)
        
        if self.args.conditional_gan:
            image_features, inputs = features

            if self.args.gen_model_type == 'transformer':
                inputs = inputs.unsqueeze(1)

                if args.gen_model_output == 'pool':
                    image_features = image_features.unsqueeze(1) 
                else:
                    image_features = image_features.view(image_features.size(0),-1,image_features.size(1))

            else:
                inputs = image_features.unsqueeze(1)
        else:
            inputs = features.unsqueeze(1)

        outputs = []
        for i in range(max_caption_len):
            
            if self.args.gen_model_type == 'lstm':
                print(inputs.shape)
                hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            else:
                if self.args.conditional_gan:
                    print("inputs, image features: ", inputs.shape, image_features.shape)
                    hiddens = self.transformer(tgt=inputs.transpose(0,1),memory=image_features.transpose(0,1)).transpose(0,1)
                else:
                    print("inputs: ", inputs.shape)
                    hiddens = self.transformer(inputs.transpose(0,1)).transpose(0,1)
            
            # print(hiddens.shape)

            if pretrain:
                pred = self.linear(hiddens)
                outputs.append(pred[:,-1])
                pred = F.softmax(pred, dim=-1)
                # print(pred.shape)
            else:
                gumbel_t = self.add_gumbel(self.linear(hiddens))            # outputs:  (batch_size, vocab_size)
                pred = F.softmax(gumbel_t * self.temperature, dim=-1)  
                # print(pred.shape)
                outputs.append(pred[:,-1])
            
            pred_index = pred.max(-1)[1]

            pred_index = pred_index[:,-1]                                     # predicted: (batch_size)          

            # print(pred_index)
            sampled_ids.append(pred_index)
            if self.args.gen_model_type == 'lstm':
                inputs = self.embed(pred_index.detach())                       # inputs: (batch_size, embed_size)
                inputs = inputs.unsqueeze(1)                                    # inputs: (batch_size, 1, embed_size)
            elif self.args.gen_model_type == 'transformer':
                inputs = torch.cat([inputs,self.embed(pred_index.detach().unsqueeze(1))],1)                    # inputs: (batch_size, n+1, embed_size)

            

        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
#         print(len(outputs),outputs[0].shape)
        outputs = torch.stack(outputs, 1)
#         print(outputs.shape)
        return outputs, sampled_ids
    
    #@staticmethod
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
        self.init_params()
        
    def forward(self, images, caps, lengths, pretrain=False):
        """Extract feature vectors from input images."""
        if self.args.cgan:
            features = self.encoder(images)
            features = [features,self.decoder.embed(torch.ones(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))]
        else:
            features = self.decoder.embed(torch.ones(len(images),1, dtype=torch.long).squeeze(1).to(self.args.device))
        pred, hidden = self.decoder(features, caps, lengths, pretrain)

        # print(pred.shape, hidden.shape)
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
        features = [generator.encoder(sample_images),generator.decoder.embed(torch.ones(b,1, dtype=torch.long).squeeze(1).to(args.device))]
        print("image features: ",generator.encoder(sample_images).shape)
    else:
        features = generator.decoder.embed(torch.ones(b,1, dtype=torch.long).squeeze(1).to(args.device))
        print("start token features: ",features.shape)

    
    pretrain_pred,_ = generator.decoder.sample(features, pretrain=True)

    adv_pred,_ = generator.decoder.sample(features)

    print(pretrain_pred.shape, adv_pred.shape)

