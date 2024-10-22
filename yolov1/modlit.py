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


def calculate_IoU(y_pred, y_hat):
    # IoU
    bbox1 = torchvision.ops.box_convert(y_pred[..., 0:4], 'xywh', 'xyxy')
    bbox2 = torchvision.ops.box_convert(y_pred[..., 5:9], 'xywh', 'xyxy')
    gtbox = torchvision.ops.box_convert(y_hat[..., 0:4], 'xywh', 'xyxy')

    bbox1 = bbox1.reshape(-1, 4)
    bbox2 = bbox2.reshape(-1, 4)
    gtbox = gtbox.reshape(-1, 4)

    iou1 = intersection_over_union(bbox1, gtbox, iou_threshold=0.5, aggregate=False)
    iou1 = torch.diagonal(iou1, 0)
    iou1 = iou1.reshape(iou1.size(0), -1)
    iou2 = intersection_over_union(bbox2, gtbox, iou_threshold=0.5, aggregate=False)
    iou2 = torch.diagonal(iou2, 0)
    iou2 = iou2.reshape(iou2.size(0), -1)
    iou = torch.cat([iou1, iou2], dim=1)

    iou_max, best_box = torch.max(iou, dim=1)
    obj_ij = y_hat[..., 4:5]

    return best_box, obj_ij


class VGG16YOLO(L.LightningModule):
    def __init__(self):
        super(VGG16YOLO, self).__init__()
        self.model = YOLO()
        self.map = MeanAveragePrecision(box_format='xywh')

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        def lr_fn(epoch):
            warm_up_epochs = 1
            if epoch < warm_up_epochs:
                return float(epoch + 1) / float(warm_up_epochs + 1)
            elif epoch < 75:
                return 10
            elif epoch < 105:
                return 1
            else:
                return 0.1
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def yolo_loss(self, y_pred, y_hat):
        # parameters
        coord = 5
        noobj = 0.5

        # Squared-sum error
        loss_fn = nn.MSELoss(reduction='sum')
        best_box, obj_ij = calculate_IoU(y_pred, y_hat)
        noobj_ij = (1 - obj_ij)

        # bbox loss
        best_box = best_box.reshape(-1, 7, 7, 1)
        bbox_pred = obj_ij*((1-best_box)*y_pred[..., 0:4] + best_box*y_pred[..., 5:9])
        bbox_pred[..., 2:4] = torch.sqrt(bbox_pred[..., 2:4])
        bbox_target = obj_ij*(y_hat[..., 0:4])
        bbox_target[..., 2:4] = torch.sqrt(bbox_target[..., 2:4])

        bbox_loss = loss_fn(torch.flatten(bbox_pred),
                            torch.flatten(bbox_target))

        # confidence loss
        obj_pred = ((1-best_box)*y_pred[..., 4:5] + best_box*y_pred[..., 9:10])
        obj_target = y_hat[..., 4:5]
        obj_loss = loss_fn(torch.flatten(obj_ij*obj_pred),
                           torch.flatten(obj_ij*obj_target))
        noobj_loss = loss_fn(torch.flatten(noobj_ij*y_pred[..., 4:5]),
                             torch.flatten(noobj_ij*obj_target))
        noobj_loss += loss_fn(torch.flatten(noobj_ij*y_pred[..., 9:10]),
                              torch.flatten(noobj_ij*obj_target))

        # class probability loss
        class_pred = y_pred[..., 10:]
        class_target = y_hat[..., 10:]
        class_loss = loss_fn(torch.flatten(obj_ij*class_pred),
                             torch.flatten(obj_ij*class_target))

        # total loss
        loss = coord*bbox_loss + noobj*noobj_loss + obj_loss + class_loss
        return loss

    def training_step(self, batch, batch_idx):
        x_tr, y_tr = batch
        hypothesis = self.model(x_tr)
        best_box, obj_ij = calculate_IoU(hypothesis, y_tr)
        best_box = best_box.reshape(-1, 7, 7, 1)
        bbox_pred = obj_ij * ((1 - best_box) * hypothesis[..., 0:4] + best_box * hypothesis[..., 5:9])
        bbox_pred = bbox_pred.reshape(-1, 4)
        loss = self.yolo_loss(hypothesis, y_tr)

        preds = [
            dict(
                boxes=bbox_pred,
                scores=hypothesis[..., 4:5].view(-1),
                labels=torch.argmax(hypothesis[..., 10:].view(-1, 20), dim=1)
            )
        ]
        target = [
            dict(
                boxes=y_tr[..., 0:4].reshape(-1, 4),
                labels=torch.argmax(y_tr[..., 10:].view(-1, 20), dim=1)
            )
        ]
        self.map.update(preds, target)
        mean_average_precision = self.map.compute()

        self.log("train loss", loss, sync_dist=True)
        self.log("mAP", mean_average_precision['map'], sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_val, y_val = batch
        hypothesis = self.model(x_val)
        best_box, obj_ij = calculate_IoU(hypothesis, y_val)
        best_box = best_box.reshape(-1, 7, 7, 1)
        bbox_pred = obj_ij * ((1 - best_box) * hypothesis[..., 0:4] + best_box * hypothesis[..., 5:9])
        bbox_pred = bbox_pred.reshape(-1, 4)
        loss = self.yolo_loss(hypothesis, y_val)
        preds = [
            dict(
                boxes=bbox_pred,
                scores=hypothesis[..., 4:5].view(-1),
                labels=torch.argmax(hypothesis[..., 10:].view(-1, 20), dim=1)
            )
        ]
        target = [
            dict(
                boxes=y_val[..., 0:4].reshape(-1, 4),
                labels=torch.argmax(y_val[..., 10:].view(-1, 20), dim=1)
            )
        ]
        self.map.update(preds, target)
        mean_average_precision = self.map.compute()

        self.log("valid loss", loss, sync_dist=True)
        self.log("mAP", mean_average_precision['map'], sync_dist=True)
