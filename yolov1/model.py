import torch
import torchvision
import torch.nn as nn
import torchvision.models.vgg as vgg
from torchvision.models import VGG16_Weights


class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()

        self.backbone = vgg.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:-1]
        self.backbone[0] = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.fc2 = nn.Linear(4096, 1470)

        for params in self.backbone.parameters():
            params.requires_grad = False

        for name, layer in self.named_modules():
            if name.find('conv') != -1 or name.find('fc') != -1:
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.normal_(layer.bias, mean=0, std=0.01)

    def forward(self, x):
        out = self.backbone(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.leaky_relu(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = out.view(-1, 7, 7, 30)

        return out


model = YOLO()
