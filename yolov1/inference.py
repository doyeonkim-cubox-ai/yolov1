import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yolov1.modlit import VGG16YOLO
import argparse
import numpy as np
import random
import cv2
from PIL import Image

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                   "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant",
                   "sheep", "sofa", "train", "tvmonitor"]
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]


def draw_box(img, cell_x, cell_y, x, y, w, h, score, label):
	height = img.shape[0] * h
	width = img.shape[1] * w
	left = (img.shape[1] * 7)*(cell_x - 1 + x) - width / 2
	top = (img.shape[0] * 7)*(cell_y - 1 + y) - height / 2

	color = colors[label]
	text = classes[label] + " " + str(float(score))
	cv2.rectangle(img, (int(left), int(top)), (int(left + width), int(top + height)), color, 2)

	text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
	p1 = (int(left), int(top) - text_size[1])

	cv2.rectangle(img,
				  (p1[0] - 2 // 2, p1[1] - 2 - baseline),
				  (p1[0] + text_size[0], p1[1] + text_size[1]), color, -1)
	cv2.putText(img, text,
				(p1[0], p1[1] + baseline),
				cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)

	return img


def draw_detection_result(img, pred, raw=False, thres=0.1):
	if raw:
		offsets = [0, 5]
	else:
		offsets = [0]

	for offset in offsets:
		pred = pred.reshape((-1, 30))
		for i in range(pred.shape[0]):
			x, y, w, h, c = pred[i][0 + offset: 5 + offset]

			# calculate cell index
			xidx = i % 7
			yidx = i // 7

			# transform cell relative coordinates to image relative coordinates
			x = (x + xidx) / 7.0
			y = (y + yidx) / 7.0

			score, idx = pred[..., 10:].max(dim=0)
			if c * score < thres: continue
			img = draw_box(img, x, y, w, h, score * iou, cat)

	return img

def draw_ground_truth(img, truth):
	"""
	Tool function to draw ground truth
	:param img: numpy image to be rendered
	:param pred: truth bbox in json format (str)
	:return: image with ground truth bbox
	"""
	pred = json.loads(truth)
	for bbox in pred:
		xmin, ymin, xmax, ymax = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
		w = xmax - xmin
		h = ymax - ymin
		x = (xmin + xmax) / 2
		y = (ymin + ymax) / 2
		img = draw_box(img, x, y, w, h, 1, bbox['category'])
	return img


def main():
	# define parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', type=str, help='Pick Model(ex. time)')
	m = parser.parse_args().model

	# load trained model
	checkpoint = f"./model/model_{m}.ckpt"
	model = VGG16YOLO.load_from_checkpoint(checkpoint, m=m)

	#########################################################################################################
	# ========================================== data preprocess ========================================== #
	#########################################################################################################
	transform = transforms.Compose([
		transforms.Resize((448, 448)),
		transforms.ToTensor()
	])

	# using your own data
	parser.add_argument('-img', type=str, help='Input Image Path')
	img_path = parser.parse_args().img
	img = Image.open(img_path)
	img_tensor = transform(img)
	img_tensor = img_tensor.unsqueeze(0).to('cuda')

	#########################################################################################################
	# ============================================== process ============================================== #
	#########################################################################################################
	# put inference img tensor into the prediction model
	result = model(img_tensor)

	#########################################################################################################
	# ============================================ postprocess ============================================ #
	#########################################################################################################
	# using your own data
	result = result.squeeze(0).detach().to('cpu').numpy()

	res = np.argmax(result[])
	print(result)
	print(classes[res])


if __name__ == '__main__':
	main()