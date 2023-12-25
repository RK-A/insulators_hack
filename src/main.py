import os
import cv2
import torch
import pandas as pd

from ultralytics import YOLO
from utils.image import ImageWithBoxes
from utils.pred import findBoxes, getPd, delete_everything_in_folder



if __name__=='__main__':

    folder = r'' # Путь до папки 


    if torch.cuda.is_available():
        torch.cuda.set_device(0)    
    model = YOLO('model/130EPH_best.pt')
    submiss = pd.DataFrame(columns=['file_name', 'rbbox', 'probability'])
    predict_folder = f'{folder}_pred'
    if not os.path.exists(predict_folder):
        os.mkdir(predict_folder)
    else: 
        delete_everything_in_folder(predict_folder)
        
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            boxes, conf = findBoxes(img, model, confidence=0.51, resize=640, f_resize=True)
            img = ImageWithBoxes(img, boxes)
            cv2.imwrite(os.path.join(predict_folder,f'pred_{filename}'), img)
            submiss = pd.concat([submiss,getPd('.'.join(filename.split('.')[:-1]),boxes, conf)],ignore_index=True)
    
    submiss.sort_values('file_name', inplace=True)
    submiss.to_csv(os.path.join(predict_folder,'submission.csv'), index =False)
