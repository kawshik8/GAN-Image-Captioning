#import config as cfg
import torch
import random
import numpy as np
from training import GANInstructor
from tasks import *#COCO_data, collate_fn
from torch.utils.data import DataLoader
from args import *

args = get_args()
# Set up random seed to 1008. Do not change the random seed.
# Yes, these are all necessary when you run experiments!

seed = 1008
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#args = get_args()
# vocab = ["<pad>", "cat","on","the","mat","hello"]
# images= torch.rand(2,3,64,64)
# lengths = torch.tensor([5,6])
# captions = torch.zeros(2,20)
# captions[0] = torch.cat((torch.tensor([2,1,3,4,5]).long(), torch.zeros(15).long()))
# captions[1] = torch.cat((torch.tensor([3,2,4,4,1,5]).long(), torch.zeros(14).long()))
# captions = captions.to(torch.long)

# train_data = [(images,captions,lengths)]

train_dataset = COCO_data(args.data_dir + "/dataset_coco.json", args.data_dir, 'train', args.image_size, args.captions_per_image, max_seq_len=args.max_seq_len, dataset_percent=args.dataset_percent)

args.vocab_size = train_dataset.vocab_size

val_dataset = COCO_data(args.data_dir + "/dataset_coco.json", args.data_dir, 'val',args.image_size, args.captions_per_image, max_seq_len=args.max_seq_len, dataset_percent=args.dataset_percent)


inst = GANInstructor(args, train_dataset, val_dataset) #Pass the validation loader for second argument. 

bleu_weights = [0.25,0.25,0.25,0.25] # 4 gram uniform weights -> BLEU-4
inst._run(bleu_weights)
#inst.evaluate(dataloader=val_data, isTest=True) #For testing still need to calculate accuracy.
