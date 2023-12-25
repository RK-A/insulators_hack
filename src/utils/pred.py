import numpy as np
import cv2
import math
import pandas as pd
import os
import shutil


def reCalc(x, i:int,j:int, s):
    w = min(x[2], s-x[0])
    h = min(x[3],s-x[1])
    return np.array([x[0]+w*0.05+s*i,x[1]+h*0.05+s*j,w*1.1,h*1.1])

def findBoxes(img, model,resize=640, f_resize=False, confidence = 0.37):
    img_shape = img.shape
    if f_resize:
        new_size = ((img_shape[1]//resize+1)*resize,(img_shape[0]//resize+1)*resize)
        img = cv2.resize(img, new_size)
        img_shape = img.shape
    tile_size = (resize, resize)
    offset = (resize, resize)


    boxes = []
    confidences = []
    for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
            cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
            # Debugging the tiles
            if len((result:=model.predict(cropped_img, conf=confidence))[0])>0:
                boxes.append(np.array(list(map(lambda x: reCalc(x, j,i,resize), result[0].boxes.xywh.cpu().numpy()))))
                confidences.append(result[0].boxes.conf.cpu().numpy())
    normalize_boxes=[]
    norm_conf = []
    for i,tile in enumerate(boxes):
        for j,box in enumerate(tile):
            x,y,w,h = box
            img_height, img_width = img.shape[:2]
            norm_conf.append(confidences[i][j])
            normalize_boxes.append([(x-w*0.1)/img_width, (y-h*0.1)/img_height, w*0.9/img_width, h*0.9/img_height])
    
    return normalize_boxes, norm_conf

def getPd(filename, data, conf):
#     function for write data
    if len(data)==0:
        return pd.DataFrame([[filename]+[[[0]*4]]+[[0]]], columns=['file_name', 'rbbox', 'probability'])
    return pd.DataFrame(np.array([filename, data.__str__(), conf.__str__()]).reshape(1,3), columns=['file_name', 'rbbox', 'probability'])


def delete_everything_in_folder(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)