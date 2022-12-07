import cv2 as cv
from torchvision.models.detection import keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights
import numpy as np
from torchvision import transforms as T

import matplotlib.pyplot as plt

def draw_keypoints_per_person(img, all_keypoints, all_scores, confs, keypoint_threshold=2, conf_threshold=0.9):
    img_copy = img.copy()
    if(len(all_keypoints) > 0):
        # initialize a set of colors from the rainbow spectrum
        cmap = plt.get_cmap('rainbow')
        # create a copy of the image
        # pick a set of N color-ids from the spectrum
        color_id = np.arange(1,255, 255//len(all_keypoints)).tolist()[::-1]
        # iterate for every person detected
        for person_id in range(len(all_keypoints)):
        # check the confidence score of the detected person
            if confs[person_id]>conf_threshold:
                # grab the keypoint-locations for the detected person
                keypoints = all_keypoints[person_id, ...]
                # grab the keypoint-scores for the keypoints
                scores = all_scores[person_id, ...]
                # iterate for every keypoint-score
                for kp in range(len(scores)):
                    # check the confidence score of detected keypoint
                    if scores[kp]>keypoint_threshold:
                        # convert the keypoint float-array to a python-list of integers
                        keypoint = tuple(map(int, keypoints[kp, :2].detach().numpy().tolist()))
                        # pick the color at the specific color-id
                        color = tuple(np.asarray(cmap(color_id[person_id])[:-1])*255)
                        # draw a circle over the keypoint location
                        cv.circle(img_copy, keypoint, 3, color, -1)
    return img_copy



torch_model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)

torch_model.eval()


cap = cv.VideoCapture(0)
transform = T.Compose([T.ToTensor()])
while(cv.waitKey(1) != 27):

    _, img = cap.read()
    img_t = transform(img)
    output = torch_model([img_t])[0]
    keypoints_img = draw_keypoints_per_person(img, output["keypoints"], output["keypoints_scores"], output["scores"], keypoint_threshold=2)

    cv.imshow("ai", keypoints_img)


