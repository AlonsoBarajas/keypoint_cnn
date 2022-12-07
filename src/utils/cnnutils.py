import torchvision
import torch
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

def saveONNX():
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    predictions = model(x)

    # optionally, if you want to export the model to ONNX:
    torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)