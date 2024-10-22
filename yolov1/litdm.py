import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset, Dataset
import lightning as L
from PIL import Image
from yolov1.data import MyData


class PascalVOC(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
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

        # if stage == "test":
        #     self.test = torchvision.datasets.VOCDetection(self.data_dir, year='2012' image_set="val",
        #                                                   transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, shuffle=False, num_workers=8)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=8)

# print("File __name__ is set to: {}" .format(__name__))
#
#
# def main():
#     transform = transforms.Compose([
#         transforms.Resize((448, 448)),
#         transforms.ToTensor()
#     ])
#     # voc2007_trainval = torchvision.datasets.VOCDetection("./pascalvoc", download=True, year='2007', image_set="trainval")
#     voc2007_train = MyData("./pascalvoc", download=True, year='2007', image_set="train", transform=transform)
#     # voc2007_test = torchvision.datasets.VOCDetection("./pascalvoc", download=True, year='2007', image_set="test")
#     # voc2012_val = torchvision.datasets.VOCDetection("./pascalvoc", download=True, year='2012', image_set="val")
#     # voc2012_train = torchvision.datasets.VOCDetection("./pascalvoc", download=True, year='2012', image_set="train")
#
#     train_loader = DataLoader(voc2007_train, batch_size=64, shuffle=False, num_workers=8)
#     # trainval_loader = DataLoader(voc2007_trainval, batch_size=64, shuffle=False, num_workers=8)
#
#     for x, y in train_loader:
#         print(x.shape)
#         print(y.shape)
#         img = transforms.ToPILImage()(x)
#         break
#
#
# if __name__ == '__main__':
#     print("main function executed")
#     main()