import config as cfg
import torch
import random
import numpy as np
from training import GANInstructor
from tasks import COCO_data, collate_fn
from torch.utils.data import DataLoader

# Set up random seed to 1008. Do not change the random seed.
# Yes, these are all necessary when you run experiments!
seed = 1008
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if cfg.cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# vocab = ["<pad>", "cat","on","the","mat","hello"]
# images= torch.rand(2,3,64,64)
# lengths = torch.tensor([5,6])
# captions = torch.zeros(2,20)
# captions[0] = torch.cat((torch.tensor([2,1,3,4,5]).long(), torch.zeros(15).long()))
# captions[1] = torch.cat((torch.tensor([3,2,4,4,1,5]).long(), torch.zeros(14).long()))
# captions = captions.to(torch.long)

# train_data = [(images,captions,lengths)]

coco_dataset = COCO_data("../../coco_data/dataset_coco.json","../../coco_data",'train')
loader = DataLoader(coco_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)


# train_data = [(images,captions,lengths)]
# val_data = [(images,captions,lengths)]
inst = GANInstructor(loader,loader)

inst._run()
#inst.evaluate(dataloader=val_data, isTest=True)
