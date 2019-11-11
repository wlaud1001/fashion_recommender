import torch
import torch.utils.data
import torchvision
import os
from PIL import Image

class DeepfashionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset[idx]['path'])
        img = Image.open(img_path).convert("RGB")
        num_objs = 1

        boxes = []
        boxes.append(self.dataset[idx]['boxes'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        labels[0] = self.dataset[idx]['labels']
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dataset)


class upper_pattern_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.dataset[idx][0])
        img = Image.open(img_path).convert("RGB")

        label = category_list.index(self.dataset[idx][1])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.dataset)