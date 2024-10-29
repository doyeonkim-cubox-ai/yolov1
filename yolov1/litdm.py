import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import lightning as L
from PIL import Image
from yolov1.data import MyData
import logging


class PascalVOC(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ColorJitter(saturation=0.15, hue=0.15, brightness=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    def prepare_data(self):
        MyData(self.data_dir, download=True, year='2007', image_set="trainval")
        MyData(self.data_dir, download=True, year='2007', image_set="test")
        MyData(self.data_dir, download=True, year='2012', image_set="val")
        MyData(self.data_dir, download=True, year='2012', image_set="train")

    def setup(self, stage: str):
        if stage == "fit":
            self.voc2007_trainval = MyData(self.data_dir, year='2007',
                                                image_set="trainval", transform=self.transform)
            self.voc2007_test = MyData(self.data_dir, year='2007',
                                                image_set="test", transform=self.transform)
            self.voc2012_train = MyData(self.data_dir, year='2012',
                                                image_set="train", transform=self.transform)
            self.voc2012_val = MyData(self.data_dir, year='2012',
                                                image_set="val", transform=self.transform)
            self.train = ConcatDataset([self.voc2007_test, self.voc2007_trainval, self.voc2012_train])
            self.valid = self.voc2012_val

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=8)
