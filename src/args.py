import torch
import os 
import torch
import argparse 

def add_model_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN Model args')

    ################### Generator ###################

    parser.add_argument('--resnet-type',
                            type=str,
                            default="resnet18",
                            choices=["resnet18","resnet34","resnet50","resnet101","resnet152"],
                            help='resnet model to use')

    parser.add_argument('--gen-hidden-dim',
                            type=int,
                            default=512,
                            help='hidden dimension of generator')

    parser.add_argument('--gen-embed-dim',
                            type=int,
                            default=512,
                            help='embedding dimension of generator')

    parser.add_argument('--gen-num-layers',
                            type=int,
                            default=6,
                            help='number of layers in generator')

    parser.add_argument('--gen-nheads',
                            type=int,
                            default=8,
                            help='number of heads for multi headed attention in generators')

    parser.add_argument('--gen-init',
                            type=str,
                            default='uniform',
                            help='Initialization strategy for generator weights')

    parser.add_argument('--gen-model-type',
                            type=str,
                            default='transformer',
                            choices=["transformer","lstm"],
                            help='type of generator to use')

    parser.add_argument('--gen-model-output',
                            type=str,
                            default='grid',
                            choices=["grid","pool"],
                            help='type of cnn output to use')

    parser.add_argument('--freeze-cnn',
                            type=int,
                            default=1,
                            choices=[0,1],
                            help='use cnn as feature extractor? or backpropogate gradients?')

    ################### Discriminator ###################

    parser.add_argument('--disc-embed-dim',
                            type=int,
                            default=64,
                            help='embeddings dimension to use in discriminator')

    parser.add_argument('--disc-num-rep',
                            type=int,
                            default=64,
                            help='number of representations to use for CNN discriminator')

    parser.add_argument('--disc-filter-sizes',
                            type=list,
                            default=[3, 4, 5],
                            help='Layer wise filter sizes to use in discriminator')

    parser.add_argument('--disc-num-filters',
                            type=list,
                            default=[300, 300, 300],
                            help='number of filters to use in discriminator per layer')

    parser.add_argument('--disc-init',
                            type=str,
                            default='uniform',
                            help='init strategy for discriminator weights')
    
    #################### Common args #####################
    
    parser.add_argument('--conditional-gan',
                            type=int,
                            default=0,
                            choices = [0,1],
                            help='is the gan conditional?')
                       
    #args = parser.parse_args()

    #return args


def add_data_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN Data args')

    ################### NLP Part ###################

    parser.add_argument('--vocab-size',
                            type=int,
                            default=100,
                            help='vocab size for training')

    parser.add_argument('--max-seq-len',
                            type=int,
                            default=34,
                            help='maximum sequence length of captions')

    parser.add_argument('--padding-idx',
                            type=int,
                            default=1,
                            help='index of padding token in vocab')

    
    ################### CV Part ###################

    parser.add_argument('--image-size',
                            type=int,
                            default=256,
                            help='resize dim of images')

    parser.add_argument('--captions-per-image',
                            type=int,
                            default=5,
                            help='no of captions to use per image')
    
    ################### Common Part ###################

    parser.add_argument('--dataset_percent',
                            type=float,
                            default=1.0,
                            help='percentage of dataset to use for training')

    parser.add_argument('--num-workers',
                            type=int,
                            default=4,
                            help='no of workers in data loader')
    #args = parser.parse_args()

    #return args

def add_training_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN training args')

    ################### Pretraining ###################

    parser.add_argument('--pretrain-lr',
                            type=float,
                            default=1e-2,
                            help='learning rate for pretraining generator')

    parser.add_argument('--pretrain-epochs',
                            type=int,
                            default=50,
                            help='number of epochs for pretraining generator')

    parser.add_argument('--pre-train-batch-size',
                            type=int,
                            default=32,
                            help='number of batches to train at each step of pretrain training')

    parser.add_argument('--pre-eval-batch-size',
                            type=int,
                            default=32,
                            help='number of batches to train at each step of pretrain evaluation')


    parser.add_argument('--pretrain-lr-patience',
                            type=int,
                            default=10,
                            help='patience for pretrain LROnPlateau scheduler')

    parser.add_argument('--pretrain-patience',
                            type=int,
                            default=10,
                            help='number of epochs to wait before early stopping')


    #################### Adversarial Training ###################

    parser.add_argument('--gen-lr',
                            type=float,
                            default=1e-4,
                            help='learning rate for adversarial training of generator')
    
    parser.add_argument('--gen-lr-patience',
                            type=int,
                            default=10,
                            help='patience for generator LROnPlateau scheduler')

    parser.add_argument('--disc-lr',
                            type=float,
                            default=1e-4,
                            help='learning rate for adversarial training of generator')

    parser.add_argument('--disc-lr-patience',
                            type=int,
                            default=10,
                            help='patience for discriminator LROnPlateau scheduler')

    parser.add_argument('--disc-train-freq',
                            type=int,
                            default=1,
                            help='ratio of training steps of disc vs gen for stabilization')

    parser.add_argument('--adv-epochs',
                            type=int,
                            default=50,
                            help='number of epochs for adversarial training')

    parser.add_argument('--adv-train-batch-size',
                            type=int,
                            default=64,
                            help='number of batches to train at each step of adversarial training')

    parser.add_argument('--adv-eval-batch-size',
                            type=int,
                            default=64,
                            help='number of batches to train at each step of adversarial evaluation')

    parser.add_argument('--adv-loss-type',
                            type=str,
                            default='standard',
                            help='Loss function to use for adversarial training')

    parser.add_argument('--advtrain-patience',
                            type=int,
                            default=10,
                            help='number of epochs to wait before early stopping')

    parser.add_argument('--temperature',
                            type=int,
                            default=100,
                            help='Temperature for rel gan training')

    parser.add_argument('--temp-adpt',
                            type=str,
                            default='exp',
                            help='Temperature adoption strategy')

    parser.add_argument('--clip-norm',
                            type=float,
                            default=5.0,
                            help='Gradient clipping threshold')

    #args = parser.parse_args()
        
    #return args

def get_args():

    parser = argparse.ArgumentParser('NLP GAN args')

    add_training_args(parser)
    add_data_args(parser)
    add_model_args(parser)
        
    parser.add_argument('--device',
                            type=str,
                            default='cuda',
                            help='device to use for training (cpu|cuda)')

    parser.add_argument('--device-ids',
                            type=int,
                            default=0,
                            help='device id (i) to use for cuda:i')

    parser.add_argument('--expt-name',
                            type=str,
                            default='debug',
                            help='Name of the experiment')

    parser.add_argument('--model-dir',
                            type=str,
                            default='models',
                            help='directory to save models')

    parser.add_argument('--data-dir',
                            type=str,
                            default='./data',
                            help='directory where data is stored')

    parser.add_argument('--save-dir',
                            type=str,
                            default='./save',
                            help='directory to save the expt logs and tensorboard logs')

    parser.add_argument('--adv-log-step',
                            type=int,
                            default=1,
                            help='Log step frequency for adversarial training')

    parser.add_argument('--pre-log-step',
                            type=int,
                            default=1,
                            help='Log step frequency for pretraining')

    parser.add_argument('--test-log-step',
                            type=int,
                            default=1,
                            help='Log step frequency for testing')

    parser.add_argument('--log-file',
                            type=str,
                            default="log",
                            help='Log file to save logs')

    parser.add_argument('--sent-log-file',
                            type=str,
                            default="sent_log",
                            help='Log file to save logs')


    args = parser.parse_args()

    expt_no = 1
    if os.path.exists(os.path.join(args.save_dir, args.expt_name + "_" + str(expt_no))):
        while os.path.exists(os.path.join(args.save_dir, args.expt_name + "_" + str(expt_no))):
            expt_no += 1
#            print(expt_no)
 
    args.expt_name = args.expt_name + "_" + str(expt_no)

    args.save_dir = os.path.join(args.save_dir, args.expt_name)
    os.mkdir(args.save_dir)
    args.model_dir = os.path.join(args.save_dir, args.model_dir)
    os.mkdir(args.model_dir)
    args.log_file = os.path.join(args.save_dir, args.log_file) 
    args.sent_log_file = os.path.join(args.save_dir, args.sent_log_file)   

    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device("cuda:0")#{args.device_ids}")
    else:
        args.device = torch.device("cpu")

    return args
