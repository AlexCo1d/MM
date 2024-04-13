import sys

import torch
import tokenizers
import torch.nn.functional as F

sys.path.append("..")
from model.archi import MM


class MyZSModel(MM):
    def __init__(self, prompts: [], pretrained_path='', device='cuda'):
        super(MyZSModel, self).__init__()
        # load weights from pretrained_path
        self.load_pretrained_weights(pretrained_path, device)
        self.tokenizer = tokenizers.Tokenizer.from_file("../mimic_wordpiece.json")
        self.tokenizer.enable_truncation(max_length=100)
        self.tokenizer.enable_padding(length=100)
        ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], []
        for prompt in prompts:
            sent = '[CLS] ' + prompt
            encoded = self.tokenizer.encode(sent)
            ids = torch.tensor(encoded.ids).unsqueeze(0)
            attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0)
            type_ids = torch.tensor(encoded.type_ids).unsqueeze(0)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
        # stack
        caption_ids = torch.stack(ids_list).squeeze().cuda()
        attention_mask = torch.stack(attention_mask_list).squeeze().cuda()
        token_type_ids = torch.stack(type_ids_list).squeeze().cuda()
        # save the text embeddings for reuse
        outputs = self.bert_encoder(latent=None, input_ids=caption_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
        self.text_embeds = F.normalize(self.text_proj(outputs.hidden_states[-1][:, 0, :]), dim=-1)

    def forward(self, image):
        """
        :param image: [B, C, H, W]
        :return: [B, Classes]
        """
        image = image.cuda()
        vision_embeds, _ = self.forward_vision_encoder(image, 0.0)
        vision_embeds = F.normalize(self.vision_proj(vision_embeds[:, 0, :]), dim=-1)
        # calculate similarity and return the logits
        return torch.matmul(vision_embeds, self.text_embeds.t())

    def load_pretrained_weights(self, path, device='cuda'):
        state_dict = torch.load(path, 'cpu')
        self.load_state_dict(state_dict['model'], strict=False)
        self.to(device)
