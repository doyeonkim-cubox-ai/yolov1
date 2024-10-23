import torch
import torchvision
import lightning as L
import torch.nn as nn
import wandb
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning.pytorch import seed_everything
from torchmetrics import Accuracy
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.detection.iou import intersection_over_union
from yolov1.model import YOLO
from yolov1.loss import YOLOLoss
from yolov1.utils import calculate_IoU, mean_average_precision


class VGG16YOLO(L.LightningModule):
    def __init__(self):
        super(VGG16YOLO, self).__init__()
        self.model = YOLO()
        self.loss_fn = YOLOLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        def lr_fn(epoch):
            warm_up_epochs = 1
            if epoch <= warm_up_epochs:
                return float(epoch + 1/9) / float(warm_up_epochs - 8/9)
            elif epoch < 75:
                return 10
            elif epoch < 105:
                return 1
            else:
                return 0.1
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x_tr, y_tr = batch
        hypothesis = self.model(x_tr)
        # loss
        loss = self.loss_fn(hypothesis, y_tr)
        # mAP
        mAP = mean_average_precision(hypothesis, y_tr, 0.5)

        self.log("training loss", loss, sync_dist=True)
        self.log("train mAP", mAP, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        hypothesis = self.model(x_val)
        # loss
        loss = self.loss_fn(hypothesis, y_val)
        # mAP
        mAP = mean_average_precision(hypothesis, y_val, 0.5)

        self.log("validation loss", loss, sync_dist=True)
        self.log("valid mAP", mAP, sync_dist=True)
