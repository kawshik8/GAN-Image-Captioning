import torchvision.transforms as transforms
import torchvision
import json 

import os
import numpy as np
import h5py
import torch
#from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
import random
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pickle

from transformers import RobertaTokenizer
random.seed(42)
class COCO_data(Dataset):
    def __init__(self, captions_path, image_path, split, image_size=256, captions_per_image=5, max_seq_len=34, dataset_percent=1.0):
        
        assert split in {'train','val','test'}

        self.split = split
        self.image_path = image_path             
        
        json_file = json.load(open(captions_path,'r'))
        
        captions = json_file['images']

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        if os.path.exists(os.path.join(image_path, split + "_" + str(captions_per_image) + ".pkl")):
            print("Loading from saved dict")

            saved_dict = pickle.load(open(os.path.join(image_path,split + "_" + str(captions_per_image) + ".pkl"),'rb'))
            self.captions = saved_dict['captions']
            # self.word_to_index = saved_dict["w2i"]
            # self.index_to_word = saved_dict["i2w"]

        else:
            # if vocab_dicts is None:
            #     self.word_to_index = {}
            #     self.index_to_word = {}
            #     self.word_to_index['<PAD>'] = 0
            #     self.word_to_index['<S>'] = 1
            #     self.word_to_index['<E>'] = 2
            #     self.word_to_index['<UNK>'] = 3
            #     self.index_to_word[0] = '<PAD>'
            #     self.index_to_word[1] = '<S>'
            #     self.index_to_word[2] = '<E>'
            #     self.index_to_word[3] = '<UNK>'
            # else:
            #     self.word_to_index, self.index_to_word = vocab_dicts
       
            print("Creating and saving dict")
            self.captions = []
            
            t_captions = captions.copy()
            with tqdm(total=len(t_captions)) as progress:
                for i,row in enumerate(t_captions):
                    #print(i)
                    if split not in row['filepath']:
                        captions.remove(t_captions[i])
                    else:
                        if split=='train':
                            caption_dict = {}
                            for key in row:
                                    caption_dict[key] = row[key]

                            self.captions.append(caption_dict)

                        else:
                            for caption in row['sentences'][:captions_per_image]:
                                caption_dict = {}
                                for key in row:
                                    if type(row[key]) != list:
                                        caption_dict[key] = row[key]
                                
                                for key in caption:
                                    caption_dict[key] = caption[key]
                                    
                                self.captions.append(caption_dict)

                        
                        # for caption in row['sentences'][:captions_per_image]:    
                        #     if vocab_dicts is None:
                        #         for word in caption['tokens']:
                        #             if word not in self.word_to_index:
                        #                 curr_len = len(list(self.word_to_index.keys()))
                        #                 self.word_to_index[word] = curr_len
                        #                 self.index_to_word[curr_len] = word

                        

                    progress.update(1)

            save_dict = {"captions":self.captions}
                
            pickle.dump(save_dict, open(os.path.join(image_path,split + "_" + str(captions_per_image) + ".pkl"),'wb+'))
                         
        self.split = split             
        self.image_size = image_size
        self.transforms = transforms.Compose(
                            [
                                transforms.Resize((self.image_size,self.image_size), interpolation=2),
                                transforms.ToTensor(), 
                                transforms.Lambda(lambda x: x.repeat(3,1,1) if x.shape[0]==1 else x),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225]),
                            ]
                        )
    
        self.vocab_size = self.tokenizer.vocab_size
        self.dataset_percent = dataset_percent
        self.max_seq_len = max_seq_len
            
                         
    def __len__(self):
        
         return int(self.dataset_percent*len(self.captions))
                         
    def reset_indices_dict(self):
        for key in self.caption_indices_dict:
            self.caption_indices_dict[key] = 0
                         
    def __getitem__(self, index):
         
#         print(index)

        caption_dict = self.captions[index]

        image_path = os.path.join(self.image_path,caption_dict['filepath'],caption_dict['filename'])

        image = Image.open(image_path)
        image = self.transforms(image)
        if self.split == "train":
            caption = random.choice(caption_dict['sentences'])['tokens']
        else:
            caption = caption_dict['tokens']

        return image, caption, torch.ones(1)*34        
           
    def collate_fn(self, batch):
        
        image_size = batch[0][0].shape[-1]

 
        max_caption_len = 0

        images = torch.zeros(len(batch),3,image_size,image_size)
        captions = []
        lengths = []
        for i in range(len(batch)):
            captions.append(batch[i][1])
            images[i] = batch[i][0]
            max_caption_len = max(max_caption_len, len(batch[i][1]))
        max_caption_len += 2

        captions = self.tokenizer.batch_encode_plus(
                captions,
                add_special_tokens=True,
                padding='longest',
                is_split_into_words=True,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors='pt',
                return_length = True
            )
        
        captions['attention_mask'] = captions['attention_mask'].squeeze().type(torch.LongTensor)
        captions['input_ids'] = captions['input_ids'].squeeze().type(torch.LongTensor)      
        captions['length'] = captions['length'].squeeze().type(torch.LongTensor)

        captions['length'], indices = torch.sort(captions['length'], dim=0, descending=True)
        captions['input_ids'] = captions['input_ids'][indices]
        captions['attention_mask'] = captions['attention_mask'][indices]

        images = images[indices]          

        return images, captions, captions['length'], captions["input_ids"].shape[1]

    def convert_to_tokens_references(self,captions, skip_special_tokens = True):
        batch_captions = []
        for cap in captions:     
            batch_captions.append([self.tokenizer.convert_ids_to_tokens(cap, skip_special_tokens = skip_special_tokens)])
        return batch_captions

    def convert_to_tokens_candidates(self,captions, skip_special_tokens = True):
        batch_captions = []
        for cap in captions:    
            batch_captions.append(self.tokenizer.convert_ids_to_tokens(cap, skip_special_tokens = skip_special_tokens))
        return batch_captions


if __name__ == '__main__':
     
    dataset = COCO_data("../coco_data/dataset_coco.json","../coco_data",'train',captions_per_image=1)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=dataset.collate_fn, num_workers=4)
    for i,instance in enumerate(loader):
        image,caption,lengths, max_len = instance
        # print("Input IDS: ", caption['input_ids'])
        # print("Attention mask: ", caption['attention_mask'])
        # print("Lengths: ", lengths, lengths.shape)
        # print("Image: ", image.shape)
        # print("Max_len", max_len)


    # print(dataset.word_to_index)
#     with tqdm(total=len(dataset)) as progress_bar:
#         for i,instance in enumerate(loader):
#             image, caption = instance

#             progress_bar.update(len(image))
            