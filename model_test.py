import os
import sys
import cv2 
import numpy as np
from typing import Tuple
from ctypes import *
from ultralytics import YOLO
import torch  
from torchvision import models, transforms  
from PIL import Image  



# def process_image(armor_panels):
#     transform = transforms.Compose([  
#             transforms.ToPILImage(),  
#             transforms.Resize((224, 224)),  # 调整为标准的 ResNet 输入大小  
#             transforms.ToTensor(),  
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],   
#                             std=[0.229, 0.224, 0.225])  
#         ])  

#     processed_panels = []  
#     for armor_panel in armor_panels:  
#         # 确保数据类型正确  
#         if isinstance(armor_panel, np.ndarray):  
#             # 确保数值范围在 0-255 之间  
#             if armor_panel.dtype != np.uint8:  
#                 armor_panel = (armor_panel * 255).astype(np.uint8)  
        
#         # 应用转换  
#         img_tensor = transform(armor_panel)  
#         processed_panels.append(img_tensor)  

#     # 将所有处理后的面板堆叠成一个批次  
#     # batch = torch.stack(processed_panels)  
#     return processed_panels 


# def modelProcessor(model_path, armor_panels):
#     model = torch.load(model_path)      
#     model.eval().to('cuda')     

#     class_map = {  
#             0: 'collection-none',  
#             1: 'collection1',  
#             2: 'collection2',  
#             3: 'collection3',  
#             4: 'collection4', 
#             5: 'collection5',
#             6: 'collection6',
#             7: 'collection7',
#             8: 'collection8',
#     }
#     results = []  
#     for i, panel in enumerate(armor_panels):  
#         print(f"Processed Panel {i} shape: {panel.shape}") 
#     armor_panels = process_image(armor_panels)
#     with torch.no_grad():  
#         for i, armor_panel in enumerate(armor_panels):  
#             armor_panel = armor_panel.unsqueeze(0)  # 添加批次维度，形状变为 (1, channels, height, width)  
#             armor_panel = armor_panel.to('cuda')  # 将张量移动到 GPU处理  
#             # 预测  
#             output = model(armor_panel)  
#             _, predicted = torch.max(output, 1)  
            
#             # 获取预测类别  
#             class_idx = predicted.item()  
#             class_name = class_map[class_idx]  
#             # image = resultDraw(image, contour_locations[i][0], contour_locations[i][1], 
#             #                              contour_locations[i][2], contour_locations[i][3], class_name)
#             results.append(class_name)  
#     print(results)
#     return image 


# if __name__ == "__main__":
#     model_path = 'dataset_pytorch_C1.pth'
#     image = cv2.imread("2174.jpg")
#     imageProcessed = process_image(image)
#     modelProcessor(model_path, imageProcessed)
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

def process_image(armor_panels):
    transform = transforms.Compose([
        transforms.ToPILImage(),  # 将 NumPy 数组或张量转换为 PIL 图像
        transforms.Resize((224, 224)),  # 调整为标准的 ResNet 输入大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 归一化
                            std=[0.229, 0.224, 0.225])
    ])

    processed_panels = []
    for armor_panel in armor_panels:
        # 确保输入是 NumPy 数组
        if isinstance(armor_panel, np.ndarray):
            # 如果图像是灰度图（单通道），转换为 RGB（3 通道）
            if len(armor_panel.shape) == 2:  # 灰度图没有通道维度
                armor_panel = cv2.cvtColor(armor_panel, cv2.COLOR_GRAY2RGB)
            # 确保数值范围在 0-255 之间
            if armor_panel.dtype != np.uint8:
                armor_panel = (armor_panel * 255).astype(np.uint8)
        # 应用转换
        img_tensor = transform(armor_panel)
        processed_panels.append(img_tensor)

    return processed_panels


def modelProcessor(model_path, armor_panels):
    # 加载模型
    model = torch.load(model_path)
    model.eval().to('cuda')  # 设置为评估模式并移动到 GPU

    # 类别映射
    class_map = {
        0: 'collection-none',
        1: 'collection1',
        2: 'collection2',
        3: 'collection3',
        4: 'collection4',
        5: 'collection5',
        6: 'collection6',
        7: 'collection7',
        8: 'collection8',
    }

    results = []
    with torch.no_grad():
        for i, armor_panel in enumerate(armor_panels):
            armor_panel = armor_panel.unsqueeze(0)  # 添加批次维度，形状变为 (1, channels, height, width)
            armor_panel = armor_panel.to('cuda')  # 将张量移动到 GPU

            # 预测
            output = model(armor_panel)
            _, predicted = torch.max(output, 1)

            # 获取预测类别
            class_idx = predicted.item()
            class_name = class_map[class_idx]
            results.append(class_name)

    print("预测结果:", results)
    return results  # 返回预测结果


if __name__ == "__main__":
    # 模型路径
    # model_path = 'armor_classification_model.pth'  #此模型没有评估模式model.eval()
    model_path = 'dataset_pytorch_C1.pth'

    # 读取图像
    image = cv2.imread("2.jpg")

    # 将图像放入列表中（因为 process_image 期望输入是列表）
    armor_panels = [image]

    # 处理图像
    processed_panels = process_image(armor_panels)

    # 使用模型进行预测
    results = modelProcessor(model_path, processed_panels)

    # 打印结果
    print("最终预测结果:", results)

