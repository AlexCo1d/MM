import re
import time
from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import tokenizers
import random

from transformers import BertTokenizer

from .randaugment import RandomAugment


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset(Dataset):
    def __init__(
            self,
            data_root,
            max_caption_length: int = 100,
            mv=False
    ):
        self.mv=mv
        self.max_caption_length = max_caption_length
        self.data_root = data_root
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        self.images_list, self.report_list = self.read_csv()
        # self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        # self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # self.tokenizer.enable_truncation(max_length=self.max_caption_length)
        # self.tokenizer.enable_padding(length=self.max_caption_length)

    def __len__(self):
        return len(self.images_list)

    def _random_mask(self, tokens):
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1] - 1):
            if masked_tokens[0][i] == 0:
                break

            if masked_tokens[0][i - 1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                masked_tokens[0][i] = 3
                continue

            if masked_tokens[0][i - 1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##':
                continue

            prob = random.random()
            if prob < 0.5:
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index):
        if self.mv:
            pass
        else:
            image = pil_loader(self.images_list[index])
            image = self.transform(image)
            sent = self.report_list[index]
            text = pre_caption(sent, self.max_caption_length)
            return {
                "image1": image,
                "text": text
            }

    def read_csv(self):
        csv_path = os.path.join(self.data_root, 'training.csv')
        df = pd.read_csv(csv_path, sep=',')
        return df["image_path"], df["report_content"]

    # def collate_fn(self, instances: List[Tuple]):
    #     image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], []
    #     # flattern
    #     for b in instances:
    #         image, ids, attention_mask, type_ids, masked_ids = b
    #         image_list.append(image)
    #         ids_list.append(ids)
    #         attention_mask_list.append(attention_mask)
    #         type_ids_list.append(type_ids)
    #         masked_ids_list.append(masked_ids)
    #
    #     # stack
    #     image_stack = torch.stack(image_list)
    #     ids_stack = torch.stack(ids_list).squeeze()
    #     attention_mask_stack = torch.stack(attention_mask_list).squeeze()
    #     type_ids_stack = torch.stack(type_ids_list).squeeze()
    #     masked_ids_stack = torch.stack(masked_ids_list).squeeze()
    #
    #     # sort and add to dictionary
    #     return_dict = {
    #         "image_1": image_stack,
    #         "labels": ids_stack,
    #         "attention_mask": attention_mask_stack,
    #         "type_ids": type_ids_stack,
    #         "ids": masked_ids_stack
    #     }
    #
    #     return return_dict

def pre_caption(caption, max_words):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

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