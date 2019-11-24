import torch
import numpy as np
import torch.utils.data
from torchvision import transforms
import matplotlib.pyplot as plt

from model import upper_category_detect_model



def upper_category_detection(img):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    upper_category_list = ['hoodie', 'sweatshirt', 'sweater', 'shirt']

    #cropping = upper_lower_detection(img)

    val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    upper_category_model_path = 'models/upper_category_model.pth'
    model = upper_category_detect_model()
    model.load_state_dict(torch.load(upper_category_model_path))
    model.to(device)


    '''img_path = os.path.join(img_path)
    img = Image.open(img_path).convert("RGB")'''

    plt.imshow(img)
    img = val_transforms(img)

    rimg = img.numpy().transpose((1,2,0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rimg = std * rimg + mean

    model.eval()

    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        # print(model_ft.state_dict())
        prediction = model(img.to(device))
        _, preds = torch.max(prediction, 1)

    return upper_category_list[preds]
