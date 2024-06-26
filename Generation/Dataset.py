import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import json
import PIL
from transformers import BertTokenizer

from Utils.randaugment import RandomAugment
from PIL import Image


class Gen_Dataset(Dataset):
    def __init__(self, data_path, transform, img_tokens=32, img_root='',
                 seq_length=512, voc_size=32000, mode='train', answer_list_flag: bool = False):

        self.data = pd.read_csv(os.path.join(data_path, f'{mode}.csv'))
        self.min_seq_length = self.data['report_content'].apply(lambda x: len(x.split())).min()
        self.max_seq_length = self.data['report_content'].apply(lambda x: len(x.split())).max()
        self.mode = mode
        self.transform = transform
        self.data_path = data_path
        self.img_root = img_root
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        text = sample['report_content']
        text = pre_caption(text)
        ##### read image pathes #####
        if self.img_root == '':
            img_path = os.path.join(sample['image_path'])
        else:
            img_path = os.path.join(self.data_path, self.img_root, sample['image_path'])
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        # final_o = self.tokenizer(pre_text, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # input_ids = final_o.input_ids
        # attention_mask = final_o.attention_mask
        # input_ids = torch.tensor(input_ids).unsqueeze(0)
        # attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        # label = self.tokenizer(Answer, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # labels_att = torch.tensor(label.attention_mask).unsqueeze(0)
        # label = torch.tensor(label.input_ids).unsqueeze(0)
        if self.mode == 'train':
            item = {
                'text_input': 'The generated report is:',
                'text_output': text,
                'image': image,
            }
        else:
            item = {
                'text_input': 'The generated report is: ',
                'text_output': text,
                'image': image,
            }

        return item




def create_dataset(args):
    dataset, data_path = args.dataset_use, args.dataset_path
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    img_size = 224
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=Image.BICUBIC),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # vqa_rad
    if dataset == 'iu':
        train_dataset = Gen_Dataset(data_path, train_transform, mode='train', img_root='images')
        test_dataset = Gen_Dataset(data_path, test_transform, mode='test', img_root='images')
    elif dataset == 'mimic':
        train_dataset = Gen_Dataset(data_path, train_transform, mode='train')
        test_dataset = Gen_Dataset(data_path, test_transform, mode='test')
    return train_dataset, test_dataset, ConcatDataset([train_dataset, test_dataset])


def pre_caption(caption, max_words=256):
    caption = re.sub(
        r"([_,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('xxxx','')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption

