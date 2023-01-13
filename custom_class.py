import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.data_dir = data_dir
        self.transforms = transforms
        # load the annotations file, it also contain information of image names
        annotations1 = json.load(open(os.path.join(data_dir, "via_project.json")))
        # print(annotations1)
        self.annotations = list(annotations1.values())        

    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.annotations[idx]["filename"]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # first id is the background, objects count from 1
        obj_ids = np.array(range(len(self.annotations[idx]["regions"]))) + 1
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []

        for i in range(num_objs):
            xmin = self.annotations[idx]["regions"][i]["shape_attributes"]["x"]
            xmax = xmin + self.annotations[idx]["regions"][i]["shape_attributes"]["width"]
            ymin = self.annotations[idx]["regions"][i]["shape_attributes"]["y"]
            ymax = ymin + self.annotations[idx]["regions"][i]["shape_attributes"]["height"]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = []
        for i in range(num_objs):
            name = self.annotations[idx]["regions"][i]["region_attributes"]["name"]
            if name == "with_mask":
                labels.append(1)
            elif name == "without_mask":
                labels.append(2)
            else:
                labels.append(0)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.annotations)