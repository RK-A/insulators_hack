import cv2

def ImageWithBoxes(img, boxes):
    for box in boxes:
        x,y,w,h = box
        dh,dw, _ = img.shape
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)
        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 4)
    return img