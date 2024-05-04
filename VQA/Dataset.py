import os
import re

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
from transformers import BertTokenizer

from Utils.randaugment import RandomAugment
from PIL import Image


class VQA_Dataset(Dataset):
    def __init__(self, data_path, transform, img_tokens=32, img_root='',
                 seq_length=512, voc_size=32000, mode='train'):
        max_caption_length = 100
        max_answer_length = 50
        with open(os.path.join(data_path, f'{mode}.json')) as f:
            self.data = json.load(f)
        self.mode = mode
        self.transform = transform
        self.data_path = data_path
        self.img_root = img_root

        # answer_list = [item['answer'] for item in self.data]
        # make it unique.
        # self.answer_list = list(dict.fromkeys(answer_list))
        # self.tokenizer.enable_padding(length=max_answer_length)
        # self.tokenizer.enable_truncation(max_length=max_answer_length)
        # self.answer_list_ids = torch.stack(
        #     [torch.tensor(self.tokenizer.encode('[CLS] ' + item + ' sep').ids) for item in answer_list])
        # self.answer_list_att = torch.stack(
        #     [torch.tensor(self.tokenizer.encode('[CLS] ' + item + ' sep').attention_mask) for item in answer_list])
        # self.tokenizer.enable_truncation(max_length=max_caption_length)
        # self.tokenizer.enable_padding(length=max_caption_length)

    def __len__(self):
        return len(self.data)

    def random_answer(self, Question, Answer):
        Answer = str(Answer)
        pre_text = 'Question: ' + Question + ' The Answer is:'
        final_o = 'Question: ' + Question + ' The Answer is:' + Answer
        return pre_text, final_o

    def __getitem__(self, idx):
        sample = self.data[idx]
        print(sample)
        Question = sample['question']
        at = 'CLOSED' if (sample['answer_type'] == 'yes' or sample['answer_type'] == 'no') else 'OPEN'
        Anwser = sample['answer']
        Question = pre_question(Question)
        Anwser = pre_answer(Anwser)

        ##### read image pathes #####
        img_path = os.path.join(self.data_path, self.img_root, sample['image_name'])
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        pre_text, final_o = self.random_answer(Question, Anwser)
        # final_o = self.tokenizer(pre_text, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # input_ids = final_o.input_ids
        # attention_mask = final_o.attention_mask
        # input_ids = torch.tensor(input_ids).unsqueeze(0)
        # attention_mask = torch.tensor(attention_mask).unsqueeze(0)

        # label = self.tokenizer(Anwser, padding='longest', truncation=True, max_length=50, return_tensors="pt")
        # labels_att = torch.tensor(label.attention_mask).unsqueeze(0)
        # label = torch.tensor(label.input_ids).unsqueeze(0)

        if self.mode == 'train':
            item =  {
                'text_input': pre_text,
                'text_output': Anwser,
                'image': image,
            }
        # some dataset don't have qid and answer_type, need to generate.
        if self.mode == 'test':
            item = {
                'text_input': pre_text,
                'text_output': Anwser,
                'image': image,
                'answer_type': at,
                'image_name': sample['image_name']
            }
        return item

    # def collate_fn_train(self, batch):
    #     input_ids = torch.stack([item['input_ids'] for item in batch])
    #     images = torch.stack([item['images'] for item in batch])
    #     labels = torch.stack([item['labels'] for item in batch])
    #     labels_att = torch.stack([item['label_att'] for item in batch])
    #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #     return {
    #         'input_ids': input_ids,
    #         'images': images,
    #         'labels': labels,
    #         'label_att': labels_att,
    #         'attention_mask': attention_mask
    #     }
    #
    # def collate_fn_test(self, batch):
    #     # ids,images,names,question, answer type, answer.
    #     input_ids = torch.stack([item['input_ids'] for item in batch])
    #     images = torch.stack([item['images'] for item in batch])
    #     image_names = [item['image_name'] for item in batch]
    #     answer_types = [item['answer_type'] for item in batch]
    #     questions = [item['question'] for item in batch]
    #     answers = [item['answer'] for item in batch]
    #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
    #     return {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'images': images,
    #         'images_name': image_names,
    #         'answer_type': answer_types,
    #         'question': questions,
    #         'answer': answers
    #     }


def create_dataset(dataset, data_path):
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
    if dataset == 'radvqa':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='VQA_RAD Image Folder')
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='VQA_RAD Image Folder')
        return train_dataset, test_dataset

    # pathvqa
    elif dataset == 'pathvqa':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='images')
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='images')
        return train_dataset, test_dataset
    # slake
    elif dataset == 'slake':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train', img_root='imgs')
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test', img_root='imgs')
        return train_dataset, test_dataset

    elif dataset == 'pmcvqa':
        train_dataset = VQA_Dataset(data_path, train_transform, mode='train')
        test_dataset = VQA_Dataset(data_path, test_transform, mode='test')
        return train_dataset, test_dataset


def pre_question(question):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace(' \t', ' ').replace('is/are', 'is').replace('near/in', 'in')
    question = question.replace('>', 'more than ').replace('-yes/no', '')
    question = question.replace('x ray', 'xray').replace('x-ray', 'xray')
    question = question.rstrip(' ')
    return question


def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    return answer
