import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from yolov1.modlit import VGG16YOLO
import argparse
from yolov1.utils import draw_box
from PIL import Image

def main():
    # define parser
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-model', type=str, help='Pick Model(ex. time)')
    # m = parser.parse_args().model

    # load trained model
    checkpoint = f"./model/10:57:24.ckpt"
    model = VGG16YOLO.load_from_checkpoint(checkpoint)

    #########################################################################################################
    # ========================================== data preprocess ========================================== #
    #########################################################################################################
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ColorJitter(saturation=0.15, hue=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # using your own data
    # parser.add_argument('-img', type=str, help='Input Image Path')
    # img_path = parser.parse_args().img
    img = Image.open('./IMG_1968.png')
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
    out = draw_box(img, result)
    out.save("out.png")


if __name__ == '__main__':
    main()
