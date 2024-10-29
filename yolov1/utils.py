import torch
import torchvision
import torch.nn as nn
from torchmetrics.functional.detection.iou import intersection_over_union
import numpy as np
import random
from PIL import ImageDraw
from PIL import Image

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]


def get_detection_boxes(pred):
    pred = pred.reshape((-1, 30))
    boxes = []
    for i in range(pred.shape[0]):
        box1 = pred[i][0:5]
        box2 = pred[i][5:10]

        # calculate cell index
        x_idx = i // 7
        y_idx = i % 7

        # transform cell relative coordinates to image relative coordinates
        box1[2] *= 448
        box2[2] *= 448
        box1[3] *= 448
        box2[3] *= 448
        box1[0] = (box1[0] + x_idx) * 64
        box1[1] = (box1[1] + y_idx) * 64
        box2[0] = (box2[0] + x_idx) * 64
        box2[1] = (box2[1] + y_idx) * 64
        box1[0] -= box1[2] / 2
        box1[1] -= box1[3] / 2
        box2[0] -= box2[2] / 2
        box2[1] -= box2[3] / 2

        box1[0:4] = torchvision.ops.box_convert(box1[0:4], 'xywh', 'xyxy')
        box2[0:4] = torchvision.ops.box_convert(box2[0:4], 'xywh', 'xyxy')

        boxes.append(box1)
        boxes.append(box2)
    return boxes


def get_detection_classes(pred):
    pred = pred.reshape(-1, 30)
    class_indexes = []
    for i in range(pred.shape[0]):
        class_indexes.append(int(torch.argmax(pred[i][10:])))

    return class_indexes


def draw_box(img, label):
    draw = ImageDraw.Draw(img)
    labels = label.reshape(-1, 30)
    boxes = get_detection_boxes(label)
    class_idx = get_detection_classes(label)
    for i in range(len(class_idx)):
        box = list(boxes[2 * i])
        if box[4] > 0.5:
            print(class_idx[i])
            draw.rectangle(box[0:4], outline=colors[class_idx[i]], width=3)
            draw.text(box[0:2], text=classes[class_idx[i]])

    return img


def calculate_IoU(box_pred, box_label):
    # preprocess
    box_pred[0:2] = [box_pred[0] - box_pred[2]/2, box_pred[1] - box_pred[3]/2]
    box_label[0:2] = [box_label[0] - box_label[2]/2, box_label[1] - box_label[3]/2]
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
    iou = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)
    iou, box_idx = torch.max(iou, dim=0)
    iou_mask = iou > threshold
    iou_mask = iou_mask.int()

    true_positive = iou_mask.sum()
    false_positive = iou_mask.size(0) - true_positive
    res.append(true_positive / (true_positive + false_positive))

    return sum(res) / len(res)


# def nms():