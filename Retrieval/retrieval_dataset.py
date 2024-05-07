import PIL
import pandas as pd
import json
import os

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
        txt_id = 0
        for img_id in range(len(self.IR_query)):
            sample=self.IR_query[img_id]
            img= PIL.Image.open(os.path.join(data_path, sample["Path"])).convert('RGB')
            img.resize((224, 224))
            self.image.append(img)
        for text_id in range(len(self.IR_query)):
            self.text.append(self.TR_query[text_id]["Text"])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img_path = os.path.join(self.data_path, sample['Path'])
        img = PIL.Image.open(img_path).convert('RGB')
        img.resize((224, 224))
        item={
            'image': img,
            'idx':idx
        }
        return item
