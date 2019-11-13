from model import get_instance_segmentation_model


from PIL import Image

import time
import os, cv2
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


def remove_background(img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    segmentation_model_path = './models/people_segmentation.pth'

    num_classes = 2

    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(segmentation_model_path))
    model.to(device)
    trans = transforms.ToTensor()
    img = trans(img)

    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    pred_mask = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()

    if prediction[0]['scores'][0] < 0.8:
        return img

    pred = np.squeeze(pred_mask)
    background = pred > 0.6 * 255
    background = np.expand_dims(background, 2)
    removed = background * img

    b, g, r = cv2.split(removed)
    rimg = cv2.merge([r, g, b])
    cv2.imshow('img', rimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return removed


'''img_name = 'sss2'
img_path = 'sample_images/'+img_name+'.jpg'

img_path = os.path.join(img_path)
img = Image.open(img_path).convert("RGB")
pred = remove_background(img)

b, g, r = cv2.split(pred)
rimg = cv2.merge([r, g, b])
cv2.imshow('img', rimg)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
