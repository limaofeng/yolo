import cv2
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LPRNet import CHARS, LPRNet
from utils.load_lpr_data import LPRDataLoader

project_path = os.getcwd()

lprnet = LPRNet(lpr_max_len=8, phase=False, class_num=len(CHARS), dropout_rate=0)
device = torch.device("cuda:0")
lprnet.to(device)
print("Successful to build network!")

# load pretrained model
state_dict = torch.load(project_path + '/runs/LPRNet__iteration_80000.pth')
# 移除 "module." 前缀
if 'module.' in next(iter(state_dict.keys())):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
lprnet.load_state_dict(state_dict)
print("load pretrained model successful!")

# 预处理图像
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # 加载图像，保持为 BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB
    img = cv2.resize(img, (94, 24))  # 调整尺寸
    img = img / 255.0  # 归一化到 [0, 1]
    img = np.transpose(img, (2, 0, 1))  # 转换为 [channels, height, width]
    img = np.expand_dims(img, axis=0)  # 增加 batch 维度
    return torch.FloatTensor(img)  # 转换为张量


def parse_output(output_tensor):
    # 取得每个位置的最大概率索引
    _, predicted_indices = torch.max(output_tensor, 1)  # shape: (batch_size, seq_length)

    # 将索引转换为字符
    predicted_indices = predicted_indices.squeeze(0).cpu().numpy()  # 转换为 numpy 数组
    recognized_plate = []

    for index in predicted_indices:
        if index < len(CHARS):  # 确保索引不越界
            recognized_plate.append(CHARS[index])

    # 去除重复字符
    # 假设在字符序列中，第一个字符可能是多个相同字符，去除后再创建字符串
    filtered_plate = ''.join(recognized_plate)
    
    # 去掉空白，返回车牌
    return filtered_plate.replace(' ', '')  # 去掉所有空格字符

# 识别车牌
def recognize_plate(image_path):
    img_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = lprnet(img_tensor)
        # 进行后续解码处理
        # 这里需要填充算法来进行字符识别解码，通常是将输出转换为字符串形式
        # prebs = output.cpu().detach().numpy()
        # preb_labels = list()
        # plate_number = prebs
        # for i in range(prebs.shape[0]):
        #     preb = prebs[i, :, :]  # 对每张图片 [68, 18]
        #     preb_label = list()
        #     for j in range(preb.shape[1]):  # 18  返回序列中每个位置最大的概率对应的字符idx  其中'-'是67
        #         preb_label.append(np.argmax(preb[:, j], axis=0))
        #     no_repeat_blank_label = list()
        #     pre_c = preb_label[0]
        #     print('plate_number---', preb_label)
        
        plate_number = parse_output(output)
        
        print('plate_number', plate_number)
    return plate_number

# 示例使用
image_path = project_path + '/datasets/ccpd-2019/rec_images/test/沪BQY511.jpg'
plate_number = recognize_plate(image_path)
print(plate_number)