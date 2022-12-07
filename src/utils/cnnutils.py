import io
import numpy as np

from torch import nn

import torch.onnx

import torch.nn as nn
import torch.nn.init as init

from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights


batch_size = 1

torch_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

torch_model.eval()

# Input to the model
x = torch.randn(batch_size, 1, 224, 224, requires_grad=False)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model, x, "keypoint_rcnn.onnx", opset_version = 11)


import onnx

onnx_model = onnx.load("keypoint_rcnn.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime

ort_session = onnxruntime.InferenceSession("keypoint_rcnn.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")