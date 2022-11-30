import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights

model = keypointrcnn_resnet50_fpn(pretrained = True)