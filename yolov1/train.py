import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import wandb
from yolov1.modlit import VGG16YOLO
from yolov1.litdm import PascalVOC
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything
import time
from torchsummary import summary


def main():
    # Fix random seed for model reproducibility
    seed_everything(777, workers=True)

    # data module
    dm = PascalVOC(data_dir="./pascalvoc", batch_size=64)

    net = VGG16YOLO()
    # summary(net, (3, 448, 448))
    # exit()
    now = time.strftime('%X')
    # print(net)
    # exit(0)

    wandb_logger = WandbLogger(log_model=False, name=f'{now}', project='yolov1')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    cp_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="validation loss",
        mode="min",
        dirpath="./model/",
        filename=f"{now}"
    )
    trainer = L.Trainer(
        max_epochs=135,
        accelerator='cuda', logger=wandb_logger,
        callbacks=[lr_monitor, cp_callback], devices=1)
    trainer.fit(net, dm)


if __name__ == "__main__":
    main()