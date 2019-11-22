import numpy as np
from PIL import Image

import time
import os
from tqdm import tqdm
import random

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
from dataset import DeepfashionDataset
from model import get_faster_rcnn
import cv2

def upper_lower_detection(img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    detection_model_path = './models/deepfashion2_model.pth'
    num_classes = 14

    model = get_faster_rcnn(num_classes)
    model.load_state_dict(torch.load(detection_model_path))
    model.to(device)


    plt.imshow(img)
    trans = transforms.ToTensor()
    img = trans(img)
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    rimg = img.mul(255).permute(1, 2, 0).byte().numpy()
    Image.fromarray(rimg)

    b, g, r = cv2.split(rimg)   # img파일을 b,g,r로 분리
    clean_img = cv2.merge([r,g,b])
    rec_img = cv2.merge([r,g,b])

    num_obj = len([x for x in prediction[0]['scores'].tolist() if x>0.75])

    cropping = {}

    for idx in range(num_obj):
         # b, r을 바꿔서 Merge

        boxes = prediction[0]['boxes'][idx]
        labels = prediction[0]['labels'][idx].tolist()
        boxes = [int(item) for item in boxes]
        #minx, miny, maxx, maxy = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])
        minx, miny, maxx, maxy = boxes[0], boxes[1], boxes[2], boxes[3]

        cropping[idx] = {'cropping': clean_img[miny:maxy, minx:maxx], 'boxes': boxes, 'label': labels}


    print(prediction)




    return cropping

    #cv2.imwrite('./results/result_4.jpg', rimg2)


