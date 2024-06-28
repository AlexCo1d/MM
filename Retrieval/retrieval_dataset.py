import pandas as pd
import json
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from Utils.pretrain_datasets import pre_caption


class retrieval_dataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data = pd.read_csv(os.path.join(data_path, 'df_200.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.data_path, sample['Path'])
        img = Image.open(img_path).convert('RGB')
        report = sample['Class']
        report = pre_caption(report, 256)
        item = {
            'image': self.transform(img),
            'text': report,
        }
        return item
