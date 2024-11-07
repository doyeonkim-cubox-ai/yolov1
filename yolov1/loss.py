import torch
import torchvision
import torch.nn as nn
from yolov1.utils import calculate_IoU, mean_average_precision


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        # parameters
        self.coord = 5
        self.noobj = 0.5

    def forward(self, pred, target):
        # bounding box
        bbox1_iou = calculate_IoU(pred[..., 0:4], target[..., 0:4])
        bbox2_iou = calculate_IoU(pred[..., 5:9], target[..., 0:4])
        ious = torch.cat([bbox1_iou.unsqueeze(0), bbox2_iou.unsqueeze(0)], dim=0)   # (2, 64, 49)
        iou, best_box = torch.max(ious, dim=0)  # (64, 49)
        best_box = best_box.reshape(-1, 7, 7, 1)
        Iobj_i = target[..., 4:5]   # (64, 7, 7, 1)

        # bbox loss
        bbox_pred = Iobj_i * ((1 - best_box) * target[..., 0:4] + best_box * target[..., 5:9])
        bbox_pred[..., 2:4] = torch.sign(bbox_pred[..., 2:4])*torch.sqrt(torch.abs(bbox_pred[..., 2:4]) + 1e-6)
        bbox_target = Iobj_i * (target[..., 0:4])
        bbox_target[..., 2:4] = torch.sqrt(bbox_target[..., 2:4])

        bbox_loss = self.mse(torch.flatten(bbox_pred, end_dim=-2),
                             torch.flatten(bbox_target, end_dim=-2))

        # confidence loss
        obj_pred = ((1 - best_box) * pred[..., 4:5] + best_box * pred[..., 9:10])
        obj_target = target[..., 4:5]
        obj_loss = self.mse(torch.flatten(Iobj_i * obj_pred),
                            torch.flatten(Iobj_i * obj_target))
        noobj_loss = self.mse(torch.flatten((1 - Iobj_i) * pred[..., 4:5], start_dim=1),
                              torch.flatten((1 - Iobj_i) * obj_target, start_dim=1))
        noobj_loss += self.mse(torch.flatten((1 - Iobj_i) * pred[..., 9:10], start_dim=1),
                               torch.flatten((1 - Iobj_i) * obj_target, start_dim=1))

        # class probability loss
        class_pred = pred[..., 10:]
        class_target = target[..., 10:]
        class_loss = self.mse(torch.flatten(Iobj_i * class_pred, end_dim=-2),
                              torch.flatten(Iobj_i * class_target, end_dim=-2))

        # total loss
        loss = self.coord * bbox_loss + self.noobj * noobj_loss + obj_loss + class_loss
        loss /= len(pred)

        return loss
