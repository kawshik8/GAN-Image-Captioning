import torch

#Data settings
vocab_size = 6
max_seq_length=20
padding_idx = 0

#Generator settings
embed_size = 100
hidden_size = 1024
num_layers = 2
gen_init='uniform'

#Disciminator settings
dis_embed_dim = 64
dis_hidden_dim = 64
num_rep = 64  # RelGAN
dis_filter_sizes = [2, 3, 4, 5]
dis_num_filters = [300, 300, 300, 300]
dis_init ='uniform'

#Train settings
ADV_train_epoch = 30
PRETRAIN_EPOCHS = 30
ADV_g_step = 1
ADV_d_step = 4
if_save = True
if_test = False
val_freq = 1

#Learning rates
gen_lr = 1e-2
gen_adv_lr = 1e-2
dis_lr = 1e-4


#Other settings
temperature = 100.0
temp_adpt='exp'
loss_type="rsgan"
clip_norm = 5.0

#Log settings
adv_log_step = 20
pre_log_step = 20
test_log_step = 1
save_root ='./logger/'
log_filename = 'log'

# Set up your device 
cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if cuda else "cpu")