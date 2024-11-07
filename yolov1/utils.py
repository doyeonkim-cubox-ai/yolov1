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
    pred = pred.reshape(-1, 30)
    for i in range(len(pred)):
        box1 = pred[i][0:4]
        box2 = pred[i][5:9]

        # calculate cell index
        x_idx = i // 7
        y_idx = i % 7

        # transform cell relative coordinates to image relative coordinates
        box1[0] = torch.sigmoid(box1[0])
        box2[0] = torch.sigmoid(box2[0])
        box1[1] = torch.sigmoid(box1[1])
        box2[1] = torch.sigmoid(box2[1])
        box1[2] = torch.sigmoid(box1[2])
        box2[2] = torch.sigmoid(box2[2])
        box1[3] = torch.sigmoid(box1[3])
        box2[3] = torch.sigmoid(box2[3])

        box1[2] *= 448
        box2[2] *= 448
        box1[3] *= 448
        box2[3] *= 448
        box1[0] = (box1[0] + x_idx) * 64
        box1[1] = (box1[1] + y_idx) * 64
        box2[0] = (box2[0] + x_idx) * 64
        box2[1] = (box2[1] + y_idx) * 64

        pred[i][0:4] = torchvision.ops.box_convert(box1[0:4], 'cxcywh', 'xyxy')
        pred[i][5:9] = torchvision.ops.box_convert(box2[0:4], 'cxcywh', 'xyxy')

    return pred


def draw_box(img, label, threshold, iou_threshold):
    draw = ImageDraw.Draw(img)
    res = get_detection_boxes(label)
    boxes = non_max_suppression(res, threshold, iou_threshold)
    print(boxes)
    for i in range(len(boxes)):
        classification = int(boxes[i][5])
        # print(box[0:5])
        box_mask1 = boxes[i][0:4] < 0
        box_mask2 = boxes[i][0:4] > 448
        box = boxes[i][0:4]
        box[box_mask1] = 0
        box[box_mask2] = 448
        box = list(box)
        draw.rectangle(box[0:4], outline=colors[classification], width=3)
        draw.text(box[0:2], text=classes[classification])

    return img


def calculate_IoU(box_pred, box_label):
    ious = []
    for i in range(len(box_pred)):
        # preprocess
        bbox = box_pred[i].reshape(-1, 4)
        gtbox = box_label[i].reshape(-1, 4)
        # IoU
        bbox = torchvision.ops.box_convert(bbox, 'cxcywh', 'xyxy')
        gtbox = torchvision.ops.box_convert(gtbox, 'cxcywh', 'xyxy')
        iou = intersection_over_union(bbox, gtbox, aggregate=False)
        iou = torch.diagonal(iou, 0)
        ious.append(iou)
    ious = torch.stack(ious, dim=0)

    return ious


def mean_average_precision(pred, label, threshold):
    res = []
    prediction = torch.flatten(pred, start_dim=1, end_dim=-2)
    target = torch.flatten(label, start_dim=1, end_dim=-2)
    for i in range(len(prediction)):
        box1 = prediction[i][..., 0:4]
        box2 = prediction[i][..., 5:9]
        gtbox = target[i][..., 0:4]
        class_pred = prediction[i][..., 10:]
        class_label = target[i][..., 10:]

        # IoU score
        iou1 = calculate_IoU(box1, gtbox)
        iou2 = calculate_IoU(box2, gtbox)
        iou = torch.cat([iou1.unsqueeze(0), iou2.unsqueeze(0)], dim=0)
        iou, box_idx = torch.max(iou, dim=0)

        # class score
        class_score = class_pred * class_label

        # true positive
        tp_mask = iou * class_score > threshold
        tp_mask = tp_mask.int()

        # true/false positive
        p_mask = iou * class_pred > threshold
        p_mask = p_mask.int()
        true_positive = tp_mask.sum()
        false_positive = p_mask.sum() - true_positive

        res.append(true_positive / (true_positive + false_positive + 1e-6))

    return sum(res) / len(res)


def non_max_suppression(bboxes, threshold, iou_threshold):
    bboxes = bboxes.reshape(-1, 30)
    box_list = []
    for i in range(len(bboxes)):
        class_score, label = torch.max(bboxes[i][10:].unsqueeze(0), dim=1)
        if bboxes[i][4] >= bboxes[i][9]:
            iou_score = bboxes[i][4]
            box_coords = bboxes[i][0:4]
        else:
            iou_score = bboxes[i][9]
            box_coords = bboxes[i][0:4]
        score = class_score*iou_score
        boxes = torch.concat([box_coords, score, label])
        if score > threshold:
            box_list.append(boxes)
            box_list.sort(key=lambda x: x[4], reverse=True)

    bboxes_after_nms = []
    while box_list:
        chosen_box = box_list.pop(0)
        box_list = [
            box for box in box_list
            if box[5] != chosen_box[5]
            or calculate_IoU(box[0:4].unsqueeze(0), chosen_box[0:4].unsqueeze(0)) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box[0:6])     # (list containing tensor(6))

    return bboxes_after_nms
