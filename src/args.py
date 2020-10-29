import torch
import os 
import torch
import argparse 

def add_model_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN Model args')

    ################### Generator ###################

    parser.add_argument('--gen-hidden-dim',
                            type=int,
                            default=1024,
                            help='hidden dimension of discriminator')

    parser.add_argument('--gen-embed-dim',
                            type=int,
                            default=100,
                            help='embedding dimension of discriminator')

    parser.add_argument('--gen-num-layers',
                            type=int,
                            default=2,
                            help='number of layers in discriminator')

    parser.add_argument('--gen-init',
                            type=str,
                            default='uniform',
                            help='Initialization strategy for generator weights')

    ################### Discriminator ###################

    parser.add_argument('--disc-hidden-dim',
                            type=int,
                            default=64,
                            help='hidden dimension of discriminator')

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
                            default=[2, 3, 4, 5],
                            help='Layer wise filter sizes to use in discriminator')

    parser.add_argument('--disc-num-filters',
                            type=list,
                            default=[300, 300, 300, 300],
                            help='number of filters to use in discriminator per layer')

    parser.add_argument('--disc-init',
                            type=str,
                            default='uniform',
                            help='init strategy for discriminator weights')
                       
    #args = parser.parse_args()

    #return args


def add_data_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN Data args')

    ################### NLP Part ###################

    parser.add_argument('--vocab-size',
                            type=int,
                            default=-1,
                            help='vocab size for training')

    parser.add_argument('--max-seq-len',
                            type=int,
                            default=20,
                            help='maximum sequence length of captions')

    parser.add_argument('--padding-idx',
                            type=int,
                            default=0,
                            help='index of padding token in vocab')

    
    ################### CV Part ###################

    parser.add_argument('--image-size',
                            type=int,
                            default=256,
                            help='resize dim of images')



    #args = parser.parse_args()

    #return args

def add_training_args(parser):

    #parser = argparse.ArgumentParser('NLP GAN training args')

    ################### Pretraining ###################

    parser.add_argument('--pretrain-lr',
                            type=float,
                            default=1e-3,
                            help='learning rate for pretraining generator')

    parser.add_argument('--pretrain-epochs',
                            type=int,
                            default=30,
                            help='number of epochs for pretraining generator')

    parser.add_argument('--pre-train-batch-size',
                            type=int,
                            default=16,
                            help='number of batches to train at each step of pretrain training')

    parser.add_argument('--pre-eval-batch-size',
                            type=int,
                            default=16,
                            help='number of batches to train at each step of pretrain evaluation')

    #################### Adversarial Training ###################

    parser.add_argument('--gen-lr',
                            type=float,
                            default=1e-3,
                            help='learning rate for adversarial training of generator')

    parser.add_argument('--disc-lr',
                            type=float,
                            default=1e-4,
                            help='learning rate for adversarial training of generator')

    parser.add_argument('--disc-train-freq',
                            type=int,
                            default=1,
                            help='ratio of training steps of disc vs gen for stabilization')

    parser.add_argument('--adv-epochs',
                            type=int,
                            default=30,
                            help='number of epochs for adversarial training')

    parser.add_argument('--adv-train-batch-size',
                            type=int,
                            default=16,
                            help='number of batches to train at each step of adversarial training')

    parser.add_argument('--adv-eval-batch-size',
                            type=int,
                            default=16,
                            help='number of batches to train at each step of adversarial evaluation')

    parser.add_argument('--adv-loss-type',
                            type=str,
                            default='rsgan',
                            help='Loss function to use for adversarial training')

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
                            default='cpu',
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
                            default=20,
                            help='Log step frequency for adversarial training')

    parser.add_argument('--pre-log-step',
                            type=int,
                            default=20,
                            help='Log step frequency for pretraining')

    parser.add_argument('--test-log-step',
                            type=int,
                            default=1,
                            help='Log step frequency for testing')

    parser.add_argument('--log-file',
                            type=str,
                            default="log",
                            help='Log file to save logs')


    args = parser.parse_args()

    expt_no = 1
    if os.path.exists(os.path.join(args.save_dir, args.expt_name + "_" + str(expt_no))):
        while os.path.exists(os.path.join(args.save_dir, args.expt_name + "_" + str(expt_no))):
            expt_no += 1
            print(expt_no)
 
    args.expt_name = args.expt_name + "_" + str(expt_no)

    args.save_dir = os.path.join(args.save_dir, args.expt_name)
    os.mkdir(args.save_dir)
    args.model_dir = os.path.join(args.save_dir, args.model_dir)
    os.mkdir(args.model_dir)
    args.log_file = os.path.join(args.save_dir, args.log_file)    

    if args.device == 'cuda' and torch.cuda.is_available():
        args.device = torch.device("cuda:0")#{args.device_ids}")
    else:
        args.device = torch.device("cpu")

    return args
