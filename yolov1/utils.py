import torch
import torchvision
import torch.nn as nn
from torchmetrics.functional.detection.iou import intersection_over_union


def calculate_IoU(box_pred, box_label):
    # IoU
    bbox = torchvision.ops.box_convert(box_pred, 'xywh', 'xyxy')
    gtbox = torchvision.ops.box_convert(box_label, 'xywh', 'xyxy')
    bbox = bbox.reshape(-1, 4)
    gtbox = gtbox.reshape(-1, 4)

    iou = intersection_over_union(bbox, gtbox, aggregate=False)
    iou = torch.diagonal(iou, 0)
    iou = iou.reshape(-1, 7, 7, 1)

    return iou


def mean_average_precision(pred, label, threshold):
    res = []
    iou1 = calculate_IoU(pred[..., 0:4], label[..., 0:4])
    iou2 = calculate_IoU(pred[..., 5:9], label[..., 0:4])
    iou = torch.cat([iou1, iou2], dim=1)
    box_idx, iou = torch.max(iou, dim=1)
    iou_mask = iou > threshold
    iou_mask = iou_mask.int()

    true_positive = iou_mask.sum()
    false_positive = iou_mask.size(0) - true_positive
    res.append(true_positive / (true_positive + false_positive))

    return sum(res) / len(res)
