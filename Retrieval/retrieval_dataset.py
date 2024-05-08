import pandas as pd
import json
import os
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class retrieval_dataset(Dataset):
    def __init__(self, data_path):
        with open(os.path.join(data_path, f'candidate.csv')) as f:
            self.data = pd.read_csv(f)
        with open(os.path.join(data_path, f'I2IR_query.csv')) as f:
            self.IR_query = pd.read_csv(f)
        with open(os.path.join(data_path, f'T2IR_query.csv')) as f:
            self.TR_query = pd.read_csv(f)
        self.data_path = data_path
        self.text = []
        self.image = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for img_id in range(len(self.IR_query)):
            sample = self.IR_query.iloc[img_id]
            img = Image.open(os.path.join(data_path, sample["Path"])).convert('RGB')
            self.image.append(self.transform(img))
        for text_id in range(len(self.TR_query)):
            self.text.append(self.TR_query.iloc[text_id]["Text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.data_path, sample['Path'])
        img = Image.open(img_path).convert('RGB')
        item = {
            'image': self.transform(img),
            'idx': idx
        }
        return item
