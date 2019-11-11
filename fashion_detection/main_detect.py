from upper_lower_detection import *
from upper_pattern_classification import *
from PIL import Image
import os, cv2
from torchvision import transforms
import numpy as np

#RGB
def get_color(img):
    data = np.reshape(img, (-1,3))
    data = np.float32(data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness,labels,centers = cv2.kmeans(data,1,None,criteria,10,flags)
    return centers[0].astype(np.int32)[::-1]


img_name = 'sss5'
img_path = 'sample_images/'+img_name+'.jpg'

img_path = os.path.join(img_path)
img = Image.open(img_path).convert("RGB")
trans = transforms.ToTensor()
img = trans(img)
img = img.mul(255).permute(1, 2, 0).byte().numpy()


b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
rimg = cv2.merge([r, g, b])


cropping = upper_lower_detection(img)
upper_categorise = [1,2]
category_list = ['short_sleeve_top', 'long_sleeve_top','short_sleeve_outwear','long_sleeve_outwear','vest','sling','shorts','trousers','skirt','short_sleeve_dress','long_sleeve_dress','vest_dress','sling_dress' ]
for idx in range(len(cropping)):
    cropped_img = cropping[idx]['cropping']
    cropped_img = Image.fromarray(cropped_img)
    boxes = cropping[idx]['boxes']
    labels = cropping[idx]['label']
    color = get_color(cropped_img)

    if cropping[idx]['label'] in upper_categorise:
        upper_pattern_pred = upper_pattern_detection(cropped_img)
        print(upper_pattern_pred)
        cv2.putText(img, upper_pattern_pred, (boxes[0]+15, boxes[1]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)


    img = cv2.rectangle(img, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (0, 255, 0), 2)

    name = category_list[labels-1]

    cv2.putText(img, name, (boxes[0], boxes[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    cv2.putText(img, str(color), (boxes[0], boxes[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)



trans = transforms.ToTensor()
img = trans(img)
img = img.mul(255).permute(1, 2, 0).byte().numpy()


b, g, r = cv2.split(img)  # img파일을 b,g,r로 분리
rimg = cv2.merge([r, g, b])

cv2.imwrite('sample_images/results/'+img_name+'.jpg', rimg)
cv2.imshow('img', rimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''img = cv2.rectangle(img, (boxes[0],boxes[1]),(boxes[2], boxes[3]), (0,255,0),2)

        cv2.putText(img, str(labels),(boxes[0],boxes[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)'''


#upper_pattern_detection(img)
