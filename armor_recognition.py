'''
-*- coding:utf-8 -*-
@file        armor recognition.py
@author      北京工业大学 PIP 战队 23 杨文远
@version     v3.8
@details

'''
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


def contourDrawing(image: np.ndarray , Color:str) -> Tuple[np.ndarray, list] :
    """Function summary : draw the contours of image
    
    Args:
        image(np.ndarray): raw image.
        
    Returns:
        np.ndarray: image of a contour around the light bar
        list: contours on image.
    
    """

    # exposure_factor = 0.621  
    # low_exposure_image = cv2.multiply(image, (1-exposure_factor))   #减小曝光度
    if Color == "blue" :
        exposure_factor = 0.621  
        low_exposure_image = cv2.multiply(image, (1-exposure_factor))   #减小曝光度
        # low_exposure_image = cv2.multiply(image, 10)
        lower_blue = np.array([30, 30, 80])#bgr
        upper_blue = np.array([256, 256, 256])
        # lower_blue = np.array([5, 5, 15])
        # upper_blue = np.array([255, 255, 255])
        hsv_image = cv2.cvtColor(low_exposure_image, cv2.COLOR_BGR2HSV)  #将低曝光度的图片从RGB颜色空间转化到HSV颜色空间
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)    #创建掩膜
        blue_objects = cv2.bitwise_and(image, image, mask=mask)  #将灯条的蓝色筛选出来
        gray = cv2.cvtColor(blue_objects, cv2.COLOR_BGR2GRAY)  #转化成灰度图
        cv2.imshow("i0",image)        
        # print(image)###############################

        # dim = image.ndim 
        # shape = image.shape
        # print("dimension is",shape)
        # condition_1 = image[:, :, 2] > 2*image[:, :, 0] + 1 #condition_1 是一个布尔数组，表示图像中所有像素的红色通道（image[:, :, 2]）是否大于蓝色通道（image[:, :, 0]）的两倍加一。
        # image[condition_1] = 0 #对于满足条件的像素，将其值设置为 0（即将这些像素的颜色变为黑色）。if R>2Blue + 1, then 0.  剔除掉绝对红色
        # # condition_2 = image[:, :, 0] < image[:, :, 1]#condition_2 是另一个布尔数组，表示图像中所有像素的红色通道（image[:, :, 2]）是否小于2倍绿色通道（image[:, :, 1]）。
        # # image[condition_2] = 0#若满足condition_2条件，将像素值设为0（即将这些像素的颜色变为黑色）。
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #转化成灰度图
        # # cv2.imshow("i2",low_exposure_image)
        cv2.imshow("i1",image)
        cv2.imshow("g1",gray)
        print("###################")
        # print(image)
    if Color == "red" :
        cv2.imshow("i0",image)        
        # print(image)###############################
        # dim = image.ndim 
        shape = image.shape
        print("dimension is",shape)
        condition_1 = image[:, :, 0] > 2 * image[:, :, 2] + 1 #condition_1 是一个布尔数组，表示图像中所有像素的蓝色通道（image[:, :, 0]）是否大于红色通道（image[:, :, 2]）的两倍加一。
        image[condition_1] = 0 #对于满足条件的像素，将其值设置为 0（即将这些像素的颜色变为黑色）。
        # condition_2 = image[:, :, 2] < 2*image[:, :, 1]#condition_2 是另一个布尔数组，表示图像中所有像素的红色通道（image[:, :, 2]）是否小于2倍绿色通道（image[:, :, 1]）。
        # image[condition_2] = 0#若满足condition_2条件，将像素值设为0（即将这些像素的颜色变为黑色）。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #转化成灰度图
        # cv2.imshow("i2",low_exposure_image)
        cv2.imshow("i1",image)
        cv2.imshow("g1",gray)
        print("###################")
        # print(image)
    ret , binary = cv2.threshold(gray,180,255,cv2.THRESH_BINARY) #二值化，进一步清晰化灯条
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
    cv2.drawContours(image, contours, -1, (2, 0, 255), 1)#将轮廓勾勒
    return image, contours 
    
def getCenter(contours: list) -> Tuple[list, list] :
    """Function summary : get each contour's center
    
    Args:
        contours(list) : contours on image.
        
    Returns:
        list: center of each contour.
        list: contours after the first screening
    
    """
    i=0
    center = []
    valid_contours = []
    for contour in contours : 
        M = cv2.moments(contour)
        if M["m00"] != 0: #图像的零阶矩不等于0，将白色的轮廓筛除
            i=i+1
            center_x = (M["m10"]/M["m00"])
            center_y = (M["m01"]/M["m00"]) #得到轮廓的中心点
            center.append([center_x,center_y])
            valid_contours.append(contour)   #将第一次筛选后的轮廓存入新列表
    return center, valid_contours

def isParallel(contours: list , center: list , image: np.ndarray) -> np.ndarray :  
    """Function summary : Sift through and match the contours on both sides of the armor,
                          then mark the armor plate with a green dot
    
    Args:
        contours(list) : contours on image.
        center(list) : center of each contour.
        image(np.ndarray) : image with contours.
        
    Returns:
        np.ndarray : image with green dot on armor.
    """
    contour_locations = []
    armor_panels = []
    for i in range(len(contours)) :  
        for j in range(i + 1, len(contours)) :  
                rect_i = cv2.minAreaRect(contours[i])  
                rect_j = cv2.minAreaRect(contours[j])  
                ratio_i1 = rect_i[1][1]/rect_i[1][0]###rect_i[1][0] is width,rect[1][1] is height
                width = rect_i[1][0]
                height = rect_i[1][1]
                ratio_j1 = rect_j[1][1]/rect_j[1][0]
                if (1/ratio_i1)>1.8 and (1/ratio_j1)>1.8 and rect_i[1][0]>=12 and rect_j[1][0]>=12 and 1.3>rect_i[1][0]/rect_j[1][0]>0.8 :
                    ##width>height，此时长边是width，短边是height,将细长的轮廓筛出，轮廓长边大于12，装甲板两侧轮廓长边之比在0.8~1.3之间 
                    angle_diff = abs(rect_i[2] - rect_j[2])
                                        ###将轮廓中心画上红圆
                    cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0, 2, 255), 10)
                    cv2.circle(image, (int(center[j][0]), int(center[j][1])), 3, (0, 2, 255), 10)
                    print( "h1, w1 ,angle1 ={%lf, %lf,%lf} h2, w2 ,angle2={%lf, %lf, %lf}"
                          %(rect_i[1][1],rect_i[1][0],rect_i[2],rect_j[1][1],rect_j[1][0],rect_j[2]))
                    ##
                    if  angle_diff<8.5 and rect_i[2]<-45 and rect_j[2]<-45 and abs(center[i][1] - center[j][1])<int(rect_j[1][0]+rect_i[1][0])*0.9 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][0])*2.9 :
                        #轮廓应向上倾斜至少45度，相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 30)
                        contour_locations.append([center_x, center_y, width, height])

                if ratio_i1>1.8 and ratio_j1>1.8 and rect_i[1][1]>=12 and rect_j[1][1]>=12 and 1.3>rect_i[1][1]/rect_j[1][1]>0.8 :
                    #height>width，长边height，短边width,将细长的轮廓筛出，轮廓长边大于12，装甲板两侧轮廓长边之比在0.8~1.3之间
                    angle_diff = abs(rect_i[2] - rect_j[2])
                                        ###将轮廓中心画上红圆
                    cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0, 2, 255), 10)
                    cv2.circle(image, (int(center[j][0]), int(center[j][1])), 3, (0, 2, 255), 10)
                    print( "h1, w1 ,angle1 ={%lf, %lf,%lf} h2, w2 ,angle2={%lf, %lf, %lf}"
                          %(rect_i[1][1],rect_i[1][0],rect_i[2],rect_j[1][1],rect_j[1][0],rect_j[2]))
                    ##
                    if angle_diff<8.5 and rect_i[2]>-45 and rect_j[2]>-45 and abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][1])*0.9 and abs(center[i][0] - center[j][0])<int(rect_i[1][1]+rect_j[1][1])*2.9:
                       #轮廓应向上倾斜至少45度，相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 30)
                        contour_locations.append([center_x, center_y, width, height])

                if (rect_i[2]>-13 and rect_j[2]<-77) and (ratio_i1)>1.8 and (1/ratio_j1)>1.8 and rect_i[1][1]>=10 and rect_j[1][0]>=10 : 
                    #灯条竖直时，有时候轮廓会识别错误，此时装甲板两侧的轮廓一条向左倾斜很小的角度，另外一条向右倾斜很小角度
                                        ###将轮廓中心画上红圆
                    cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0, 2, 255), 10)
                    cv2.circle(image, (int(center[j][0]), int(center[j][1])), 3, (0, 2, 255), 10)
                    print( "h1, w1 ,angle1 ={%lf, %lf,%lf} h2, w2 ,angle2={%lf, %lf, %lf}"
                          %(rect_i[1][1],rect_i[1][0],rect_i[2],rect_j[1][1],rect_j[1][0],rect_j[2]))
                    ##
                    if abs(center[i][1] - center[j][1])<int(rect_i[1][1]+rect_j[1][0])*0.9 and abs(center[i][0] - center[j][0])<int(rect_i[1][1]+rect_j[1][0])*2.9:
                        #相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 30)
                        contour_locations.append([center_x, center_y, width, height])

                if (rect_j[2]>-13 and rect_i[2]<-77) and (ratio_j1)>1.8 and (1/ratio_i1>1.8) and rect_j[1][1]>=10 and rect_i[1][0]>=10 :
                    #灯条竖直时，有时候轮廓会识别错误，此时装甲板两侧的轮廓一条向左倾斜很小的角度，另外一条向右倾斜很小角度
                                        ###将轮廓中心画上红圆
                    cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0, 2, 255), 10)
                    cv2.circle(image, (int(center[j][0]), int(center[j][1])), 3, (0, 2, 255), 10)
                    print( "h1, w1 ,angle1 ={%lf, %lf,%lf} h2, w2 ,angle2={%lf, %lf, %lf}"
                          %(rect_i[1][1],rect_i[1][0],rect_i[2],rect_j[1][1],rect_j[1][0],rect_j[2]))
                    ##
                    if abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][0])*0.9 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][1])*2.9:  
                        #相邻轮廓中心点纵坐标之差小于灯条长度和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 30)
                        contour_locations.append([center_x, center_y, width, height])

                if (rect_j[2]==rect_i[2]==-90) and (1/ratio_j1)>1.8 and (1/ratio_i1>1.8) and rect_j[1][0]>=10 and rect_i[1][0]>=10:
                    #两个灯条均垂直时
                                        ###将轮廓中心画上红圆
                    cv2.circle(image, (int(center[i][0]), int(center[i][1])), 3, (0, 2, 255), 10)
                    cv2.circle(image, (int(center[j][0]), int(center[j][1])), 3, (0, 2, 255), 10)
                    print( "h1, w1 ,angle1 ={%lf, %lf,%lf} h2, w2 ,angle2={%lf, %lf, %lf}"
                          %(rect_i[1][1],rect_i[1][0],rect_i[2],rect_j[1][1],rect_j[1][0],rect_j[2]))
                    ##                    
                    if abs(center[i][1] - center[j][1])<int(rect_j[1][1]+rect_i[1][0])*0.9 and abs(center[i][0] - center[j][0])<int(rect_i[1][0]+rect_j[1][1])*2.9:   
                        #相邻轮廓中心点纵坐标之差小于灯条长度之和的0.2倍，中心点横坐标之差小于灯条长度之和的2倍
                        center_x = int((center[i][0] + center[j][0]) / 2)  
                        center_y = int((center[i][1] + center[j][1]) / 2)  
                        cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), 30)
                        contour_locations.append([center_x, center_y, width, height])
    print(f'--------{contour_locations}')
    if not contour_locations:  
        print("轮廓列表为空，无法提取护甲面板。")  
    else :
        print("轮廓不为空··")
    armor_panels, contour_locations, image = getArmor(image, contour_locations)
    for i, panel in enumerate(armor_panels):  
        cv2.imshow(f"Armor Panel {i}", panel)  
    cv2.waitKey(0)  
    cv2.destroyAllWindows()  
    print("armor is ",armor_panels)
    print(f"-----------------------armor_panels length{len(armor_panels)}")
    return image, armor_panels, contour_locations

def getArmor(image: np.ndarray, contour_locations: list) -> list:
    armor_panels = []  
    for contour_location in contour_locations:  
        x = contour_location[0]
        y = contour_location[1]
        w = contour_location[2]
        h = contour_location[3]
        # print(f"x: {x}, y: {y}, w: {w}, h: {h}")  
        # print(f"y + h/2: {int(y + h/2)}, y - h/2: {int(y - h/2)}, x - w/2: {int(x - w/2)}, x + w/2: {int(x + w/2)}") 

        # if (y - h/2 < 0 or y + h/2 > image.shape[0] or  
        #     x - w/2 < 0 or x + w/2 > image.shape[1]):  
        #     print(f"裁剪区域超出图像边界，跳过轮廓 {contour}.")  
        #     continue
        # else:
        #     print("未超出边界")

        armor_panel = image[int(y - 1.5*h):int(y + 1.5*h), int(x - 1.5*h):int(x + 1.5*h)]  #将装甲板截取
        armor_panels.append(armor_panel)    
    print(f"armorpanel类型={type(armor_panel)}")
    print(f"armor_panels length{len(armor_panels)}")
    # for i, panel in enumerate(armor_panels):  
    #     cv2.imshow(f"Armor Panel {i}", panel)  
    # cv2.waitKey(0)  
    # cv2.destroyAllWindows()  

    return armor_panels, contour_locations, image

# def YOLOProcessor(YOLO_path, image, contour_locations, armor_panels):
#     model = YOLO(YOLO_path)
#     temp_dir = 'temp_crops'
#     os.makedirs(temp_dir, exist_ok=True)  
#     print(f"armor_panels length{len(armor_panels)}")
#     for index, armor_panel in enumerate(armor_panels):
#         try:
#             temp_path = os.path.join(temp_dir, f'crop_{index}.jpg')  
#             cv2.imwrite(temp_path, armor_panel)  
#             model(temp_path, show = True, save = False)
#             print("ok......")
#         except Exception as e:  
#             print(f"Error processing region {index}: {e}")  
#             continue  
def process_image(armor_panels):
    transform = transforms.Compose([  
            transforms.ToPILImage(),  
            transforms.Resize((224, 224)),  # 调整为标准的 ResNet 输入大小  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   
                            std=[0.229, 0.224, 0.225])  
        ])  

    processed_panels = []  
    for armor_panel in armor_panels:  
        # 确保数据类型正确  
        if isinstance(armor_panel, np.ndarray):  
            # 确保数值范围在 0-255 之间  
            if armor_panel.dtype != np.uint8:  
                armor_panel = (armor_panel * 255).astype(np.uint8)  
        
        # 应用转换  
        img_tensor = transform(armor_panel)  
        processed_panels.append(img_tensor)  

    # 将所有处理后的面板堆叠成一个批次  
    # batch = torch.stack(processed_panels)  
    return processed_panels 

def resultDraw(image, x, y, w, h, class_name,   
                color=(0, 255, 0), thickness=2):  
    """  
    在图像上绘制结果  
    """  
    # 绘制矩形框  
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cv2.rectangle(image, (x - 2*h, y - 2*h), (x + 2*h, y + 2*h), color, thickness)  
    
    # 准备标签文本  
    label = f'{class_name} '  
    
    # 获取文本大小  
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  
    
    # 绘制标签背景  
    cv2.rectangle(image, (x, y - label_h - 10), (x + label_w, y), color, -1)  
    
    # 绘制标签文本  
    cv2.putText(image, label, (x, y - 5),  
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  

def modelProcessor(model_path, image, contour_locations, armor_panels):
    model = torch.load(model_path)      
    model.eval().to('cuda')     

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
    for i, panel in enumerate(armor_panels):  
        print(f"Processed Panel {i} shape: {panel.shape}") 
    armor_panels = process_image(armor_panels)
    with torch.no_grad():  
        for i, armor_panel in enumerate(armor_panels):  
            armor_panel = armor_panel.unsqueeze(0)  # 添加批次维度，形状变为 (1, channels, height, width)  
            armor_panel = armor_panel.to('cuda')  # 将张量移动到 GPU处理  
            # 预测  
            output = model(armor_panel)  
            _, predicted = torch.max(output, 1)  
            
            # 获取预测类别  
            class_idx = predicted.item()  
            class_name = class_map[class_idx]  
            image = resultDraw(image, contour_locations[i][0], contour_locations[i][1], 
                                         contour_locations[i][2], contour_locations[i][3], class_name)
            results.append(class_name)  
    print(results)
    return image 
    





# class YOLOProcessor:  
#     def __init__(self, model_path):  
#         """  
#         初始化 YOLO 处理器  
#         :param model_path: YOLO 模型路径  
#         """  
#         self.model = YOLO(model_path)  
#         self.temp_dir = 'temp_crops'  
#         os.makedirs(self.temp_dir, exist_ok=True)  

#     def process_multiple_regions(self, image, contour_locations, armor_panels):  
#         """  
#         处理图像中的多个区域  
#         :return: 处理后的图像  
#         """  
         
#         if image is None:  
#             print("Error: Could not read image")  
#             return None  

#         # 创建结果列表  
#         results_list = []  

#         # 处理每个区域  
#         # for i, (x, y, w, h) in enumerate(regions):  
#         for index, armor_panel in enumerate(armor_panels):
#             # 截取区域  
#             try:   

#                 # 保存临时文件  
#                 temp_path = os.path.join(self.temp_dir, f'crop_{index}.jpg')  
#                 cv2.imwrite(temp_path, armor_panel)  

#                 # 预测  
#                 results = self.model(temp_path)  
                
#                 # 处理预测结果  
#                 for result in results:  
#                     if len(result.boxes) > 0:  
#                         # 获取最高置信度的预测  
#                         confidence = result.boxes.conf[0]  
#                         class_id = result.boxes.cls[0]  
#                         class_name = self.model.names[int(class_id)]  

#                         # 存储结果  
#                         results_list.append({  
#                             'region_id': index,  
#                             'coords': (contour_locations[index, 0], contour_locations[index, 1],
#                                         contour_locations[index, 2], contour_locations[index, 3]),  
#                             'class_name': class_name,  
#                             'confidence': confidence  
#                         })  

#                         # 在图像上标注结果  
#                         self._draw_result(image, contour_locations[index, 0], contour_locations[index, 1], 
#                                           contour_locations[index, 3], contour_locations[index, 4], class_name, confidence)  

#             except Exception as e:  
#                 print(f"Error processing region {index}: {e}")  
#                 continue  

#             # finally:  
#             #     # 清理临时文件  
#             #     if os.path.exists(temp_path):  
#             #         os.remove(temp_path)  

#         return image, results_list  

#     def _draw_result(self, image, x, y, w, h, class_name, confidence,   
#                     color=(0, 255, 0), thickness=2):  
#         """  
#         在图像上绘制结果  
#         """  
#         # 绘制矩形框  
#         cv2.rectangle(image, (x, y), (x + 2*h, y + 2*h), color, thickness)  
        
#         # 准备标签文本  
#         label = f'{class_name} {confidence:.2f}'  
        
#         # 获取文本大小  
#         (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)  
        
#         # 绘制标签背景  
#         cv2.rectangle(image, (x, y - label_h - 10), (x + label_w, y), color, -1)  
        
#         # 绘制标签文本  
#         cv2.putText(image, label, (x, y - 5),   
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)  

#     def display_results(self, image, wait_key=0):  
#         """  
#         显示处理后的图像  
#         """  
#         cv2.imshow('Results', image)  
#         cv2.waitKey(wait_key)  
#         cv2.destroyAllWindows()  

# def classifyArmor(YOLO_path, image, contour_locations, armor_panels):  
#     # 初始化处理器  
#     processor = YOLOProcessor(YOLO_path)   

#     # 处理图像   
#     processed_image, results = processor.process_multiple_regions(image, contour_locations, armor_panels)  

#     # 显示结果  
#     # if processed_image is not None:  
#     #     processor.display_results(processed_image)  

#     #     # 打印详细结果  
#     #     print("\nDetection Results:")  
#     #     for result in results:  
#     #         print(f"Region {result['region_id']}: "  
#     #               f"Class: {result['class_name']}, "  
#     #               f"Confidence: {result['confidence']:.2f}") 
#     return processed_image




def imageFilter(model_path: str, image: np.ndarray , Color: str) -> np.ndarray :
    """Function summary : process each frame by the function above
    
    Args:
        image(np.ndarray) : raw image.
        
    Returns:
        np.ndarray : image with green dot on armor.
    """
    image, contours = contourDrawing(image, Color)
    if contours is None or len(contours) == 0:  
        return image    
    center = []
    valid_contours = []
    center, valid_contours = getCenter(contours)
    image, armor_panels, contour_locations = isParallel(valid_contours, center,image)
    # image = classifyArmor(YOLO_path, image, contour_locations, armor_panels)
    modelProcessor(model_path, image, contour_locations, armor_panels)
    return image

     


if __name__ == "__main__" :
    model_path = 'dataset_pytorch_C1.pth'
    # video_capture = cv2.VideoCapture('test.mp4')  #打开视频，名字test.mp4
    # Color = input("red or blue?:")
    # if not video_capture.isOpened():  
    #     print("Error: Could not open video.")  
    #     exit()    
    # while True:  
    #     ret, frame = video_capture.read()  
    #     if not ret:  
    #         break  
    #     img = imageFilter(frame , Color) 
    #     cv2.imshow("img",img)        
    #     if cv2.waitKey(1) == ord('q'):  
    #         break                          
    # video_capture.release()  
    # cv2.destroyAllWindows()
    # Color = "red"
    Color = "blue"
    # image = cv2.imread("506.jpg")
    # image = cv2.imread("2174.jpg")
    # image = cv2.imread("809.jpg")
    image = cv2.imread("1.jpg")
    if image is None :
        print("empty")
    else : 
        print("not empty")
    image = imageFilter(model_path, image, Color)
    cv2.imshow("image",image)
    cv2.waitKey()

    



