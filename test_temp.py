'''
only for testing

-- alexyhzou
'''
import re

# import torch
# from functools import partial
# import torch.nn as nn
# from model.archi import MM
# # 假设的输入维度
# batch_size = 2
# img_size = 448  # 图像大小
# in_chans = 3  # 输入通道数
# num_patches = 196  # 假设patch大小为16
# text_length=100 # 假设文本长度为100
# # 创建模拟输入数据
# fake_images = torch.rand(batch_size, in_chans, img_size, img_size)  # 模拟图像数据
# fake_ids = torch.randint(0, 1000, (batch_size, text_length)).long()  # 模拟文本ids
# fake_labels = torch.randint(0, 2, (batch_size,text_length)).long()  # 模拟标签
# fake_attention_mask = torch.ones(batch_size, text_length)  # 全1的attention mask
# fake_type_ids = torch.zeros(batch_size, text_length).long()  # 假设全部是第一类型的token
#
# # 将模拟数据打包成字典，模拟实际使用中的数据批次
# batch = {
#     "image_1": fake_images,
#     "ids": fake_ids,
#     "labels": fake_labels,
#     "attention_mask": fake_attention_mask,
#     "type_ids": fake_type_ids
# }
#
# # 实例化模型
# model = MM(patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True,local_contrastive_loss=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#
# # For a model
# model = model.to(device)
#
# # 将模型转换为评估模式（这对于某些模块如Dropout和BatchNorm很重要）
# model.eval()
#
# # 前向传播
# with torch.no_grad():  # 不计算梯度，减少内存/计算需求
#     output = model(batch)
#
# # 检查输出
# print(output[1].shape)
# print(output[2].shape)

#
# path_to_pth = r'C:\Users\Alex\Downloads\MRM.pth'  # 请替换为你的.pth文件的实际路径
# model_weights = torch.load(path_to_pth,map_location=torch.device('cpu'))
# for key in model_weights['model'].keys():
#     print(key)
#     print(model_weights['model'][key].shape)
#
# model.load_state_dict(model_weights['model'], strict=False)
#
# import os
# import csv
# #
# # Paths to your directories (adjust as necessary)
# base_dir = '/home/data/Jingkai/alex/mimic/files'
# # Path for the output CSV file
# output_csv_path = '/home/data/Jingkai/alex/mimic/training.csv'
#
#
# def find_final_report(content):
#     # Search for the start of the final report
#     start_index = content.find('FINAL REPORT')
#     if start_index != -1:
#         # Return the content from 'FINAL REPORT' onwards
#         return content[start_index:]
#     else:
#         # If 'FINAL REPORT' not found, return None or empty string
#         return None
#
#
# # Open the CSV file for writing
# with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     # Write the header row
#     writer.writerow(['image_path', 'report_content'])
#
#     # Walk through the directory
#     for root, dirs, files in os.walk(base_dir):
#         for file_name in files:
#             if file_name.endswith('.jpg'):
#                 # Construct the full path to the image
#                 image_path = os.path.join(root, file_name)
#
#                 # Change the extension from .jpg to .txt to find the corresponding report
#                 report_filename = os.path.split(image_path)[0] + '.txt'
#                 report_path = report_filename
#                 # Read the report content
#                 try:
#                     with open(report_path, 'r', encoding='utf-8') as report_file:
#                         report_content = report_file.read()
#                         # Find and extract 'FINAL REPORT' content
#                         final_report_content = find_final_report(report_content)
#                         if final_report_content:
#                             # Replace newlines with spaces
#                             final_report_content = final_report_content.replace('\n', ' ').strip()
#                             # Write the image path and processed report content to the CSV
#                             writer.writerow([image_path, final_report_content])
#                         else:
#                             print(f"'FINAL REPORT' not found in: {report_filename}")
#
#                 except FileNotFoundError:
#                     print(f"Report file not found for image: {file_name}")
#
# print("CSV file has been created.")
# import pandas as pd
# df = pd.read_csv(output_csv_path)
# row_count = df.shape[0]
# print(f"CSV 文件的行数为：{row_count}")
# import torch
#
# from VQA.model_VQA import MyVQAModel

# t=torch.load('/home/data/Jingkai/alex/pretrain0/checkpoint-40.pth', map_location='cpu')
# u={}
# u['model']=t['model']
# torch.save(u,'/home/data/Jingkai/alex/weight/MM1.pth')

# from PIL import Image
# import pathlib
# from concurrent.futures import ProcessPoolExecutor
# import time
#
# def resize_image(image_path):
#     """
#     Resize the given image to 448x448, apply grayscale, and measure the time taken.
#     """
#     start_time = time.time()  # 开始计时
#
#     with Image.open(image_path) as img:
#         # 应用RandomResizedCrop等效操作
#         img = img.resize((448, 448), Image.BICUBIC)  # 等效于RandomResizedCrop
#         img = img.convert('L').convert('RGB')  # 等效于Grayscale(num_output_channels=3)
#         img.save(image_path)
#
#     end_time = time.time()  # 结束计时
#     print(f"Processed {image_path.name} in {end_time - start_time:.4f} seconds.")
#
# def main(directory_path):
#     """
#     Recursively traverse the directory, find all JPG images,
#     and resize them in parallel while measuring time.
#     """
#     path = pathlib.Path(directory_path)
#     jpg_images = list(path.glob('**/*.jpg'))
#
#     with ProcessPoolExecutor() as executor:
#         executor.map(resize_image, jpg_images)
#
# main('/mnt/data/yueli/files')

# batch_size = 2
# seq_length = 100
# vocab_size = 1000
# hidden_dim = 768
# # 生成随机的问题和答案的ids和attention mask
# images=torch.rand(batch_size, 3, 448, 448)
# input_ids = torch.randint(low=1, high=vocab_size, size=(batch_size, seq_length))
# attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
# answer_ids = torch.randint(low=1, high=vocab_size, size=(100, seq_length))  # +1是为了bos token
# answer_attention = torch.ones(100, seq_length, dtype=torch.long)
#
# # 由于`rank_answer`函数需要模型的一些内部状态，我们在这里不直接调用它。
# # 下面是如何在一个假设的模型中使用这些输入的示例。
# model=MyVQAModel()
# topk_ids, topk_probs = model(images, input_ids, attention_mask, answer_ids, answer_attention, train=False)
#
# print(f"Topk IDs shape: {topk_ids.shape}, Topk Probs shape: {topk_probs.shape}")
# print   (topk_ids)
# print(topk_probs)
# _, pred_idx = topk_probs[0].max(dim=0)
# i=topk_ids[0][pred_idx]
# print(i)

# from model.archi_Former import MM_Former
# import torch
# from model.archi import MM
# from functools import partial
# import torch.nn as nn
# fake_images = torch.rand(2, 3, 448, 448)  # 模拟图像数据
# text= ['sadflj123','231']
# batch= {
#     'image1': fake_images,
#     'text': text
#
# }
# model = MM_Former(patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
#         decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
#         mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True, local_contrastive_loss=True)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # For a model
# model = model.to(device)
#
# # 将模型转换为评估模式（这对于某些模块如Dropout和BatchNorm很重要）
# model.eval()
#
# # 前向传播
# with torch.no_grad():  # 不计算梯度，减少内存/计算需求
#     output = model(batch)
#
# # 检查输出
# print(output)

# import pandas as pd
# import csv
# import os
#
# # 读取CSV文件
# meta = pd.read_csv('./mimic-cxr-2.0.0-metadata.csv', sep=',')
# train = pd.read_csv('./training.csv', sep=',')
# folder_path = "/home/data/Jingkai/alex/mimic/files"
# with open('./training_mv.csv', 'w') as f:
#     writer = csv.writer(f)
#     writer.writerow(['study_id', 'image_path', 'view_type', 'report_content'])
#     for root, dirs, files in os.walk(folder_path):
#         # files is list of files, root is current full dir.
#         if root.split('/')[-1].startswith('s'):
#             if len(files) > 1:
#                 with open(root + '.txt', 'r') as t:
#                     study_id = root.split('/')[-1].replace("s", "")
#                     image_path = []
#                     view_type = []
#                     for filename in files:
#                         dicom_id = filename.split('.')[0]
#                         image_path.append(os.path.join(root, filename))
#                         type=str(meta[meta['dicom_id'] == dicom_id]['ViewPosition'].values[0])
#                         if type!='nan':
#                             view_type.append(type)
#                     report_content = t.read()
#                     report_content.replace('\n','')
#                     report_content = report_content.replace('\n', ' ')
#                     # 移除多余的空格
#                     report_content = re.sub(r'\s+', ' ', report_content)
#                     image_path = ';'.join(image_path)
#                     view_type = ';'.join(view_type)
#                     writer.writerow([study_id, image_path, view_type, report_content])

import re


def extract_sections(report):
    # This regular expression looks for the sections FINDINGS and IMPRESSION
    # and extracts all text up to the next all-caps word or the end of the string.
    pattern = r"(FINDINGS:.*?)(?=\n[A-Z]+:|$)|(IMPRESSION:.*?)(?=\n[A-Z]+:|$)"

    extracted_text = ''

    # Searching the report using the pattern
    matches = re.findall(pattern, report, re.DOTALL)

    # Each match contains tuples with the content of the sections
    for match in matches:
        if match[0].startswith('FINDINGS'):
            extracted_text += match[0] + ' '
        elif match[1].startswith('IMPRESSION'):
            extracted_text += match[1] + ' '

    return extracted_text.strip()


import pandas as pd

df = pd.read_csv('/home/data/Jingkai/alex/mimic/training.csv', sep=',')
df['report_content'] = df['report_content'].apply(extract_sections)
df.to_csv('/home/data/Jingkai/alex/mimic/training.csv', index=False)

df = pd.read_csv('/home/data/Jingkai/alex/mimic/training.csv', sep=',')
df = df[df['report_content'].notna()]
df.to_csv('/home/data/Jingkai/alex/mimic/training.csv', index=False)