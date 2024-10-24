import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
from torchvision.datasets import VOCDetection
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                   "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]


class MyData(VOCDetection):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
        label = torch.zeros([7, 7, 30], dtype=torch.float64)
        img_width = float(target['annotation']['size']['width'])
        img_height = float(target['annotation']['size']['height'])
        for obj in target['annotation']['object']:
            class_idx = classes.index(obj['name'])
            x_min = float(obj['bndbox']['xmin']) * (448 / img_width)
            x_max = float(obj['bndbox']['xmax']) * (448 / img_width)
            y_min = float(obj['bndbox']['ymin']) * (448 / img_height)
            y_max = float(obj['bndbox']['ymax']) * (448 / img_height)
            bbox_x = (((x_min + x_max) / 2) % 64) / 64
            bbox_y = (((y_min + y_max) / 2) % 64) / 64
            bbox_width = (x_max - x_min) / 448
            bbox_height = (y_max - y_min) / 448
            cell_x = int(((x_min + x_max) / 2) / 64)
            cell_y = int(((x_min + x_max) / 2) / 64)
            if label[cell_x][cell_y][4] == 0.0:
                label[cell_x][cell_y][0] = bbox_x
                label[cell_x][cell_y][1] = bbox_y
                label[cell_x][cell_y][2] = bbox_width
                label[cell_x][cell_y][3] = bbox_height
                label[cell_x][cell_y][4] = 1
                label[cell_x][cell_y][10 + class_idx] = 1

        if self.transform is not None:
            img = self.transform(img)

        return img, label