import cv2
import numpy as np
 
onnx_model_path = "./keypoint_rcnn.onnx" 
sample_image = ""
 
#The Magic:
net =  cv2.dnn.readNetFromONNX(onnx_model_path) 
image = cv2.imread(sample_image)
blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (224, 224),(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
preds = net.forward()
biggest_pred_index = np.array(preds)[0].argmax()
print ("Predicted class:",biggest_pred_index)
 
import requests
 
LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
labels = {int(key):value for (key, value)
          in requests.get(LABELS_URL).json().items()}
 
print("The class",biggest_pred_index, "correspond to", labels[biggest_pred_index])