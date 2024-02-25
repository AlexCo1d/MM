'''
only for testing

-- alexyhzou
'''

import torch
from functools import partial
import torch.nn as nn
from model.architecture import MM

# 假设的输入维度
batch_size = 2
img_size = 448  # 图像大小
in_chans = 3  # 输入通道数
num_patches = 196  # 假设patch大小为16
text_length=100 # 假设文本长度为100
# 创建模拟输入数据
fake_images = torch.rand(batch_size, in_chans, img_size, img_size)  # 模拟图像数据
fake_ids = torch.randint(0, 1000, (batch_size, text_length)).long()  # 模拟文本ids
fake_labels = torch.randint(0, 2, (batch_size,text_length)).long()  # 模拟标签
fake_attention_mask = torch.ones(batch_size, text_length)  # 全1的attention mask
fake_type_ids = torch.zeros(batch_size, text_length).long()  # 假设全部是第一类型的token

# 将模拟数据打包成字典，模拟实际使用中的数据批次
batch = {
    "image_1": fake_images,
    "ids": fake_ids,
    "labels": fake_labels,
    "attention_mask": fake_attention_mask,
    "type_ids": fake_type_ids
}

# 实例化模型
model = MM(patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For a model
model = model.to(device)

# 将模型转换为评估模式（这对于某些模块如Dropout和BatchNorm很重要）
model.eval()

# 前向传播
with torch.no_grad():  # 不计算梯度，减少内存/计算需求
    output = model(batch)

# 检查输出
print(output[1].shape)
print(output[2].shape)
#
# path_to_pth = r'C:\Users\Alex\Downloads\MRM.pth'  # 请替换为你的.pth文件的实际路径
# model_weights = torch.load(path_to_pth,map_location=torch.device('cpu'))
# for key in model_weights['model'].keys():
#     print(key)
#     print(model_weights['model'][key].shape)
#
# model.load_state_dict(model_weights['model'], strict=False)
