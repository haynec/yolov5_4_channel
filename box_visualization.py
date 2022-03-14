from distutils import dep_util
import cv2
from cv2 import line
import torch
import numpy as np
import os
import glob

from utils.plots import Annotator, colors, save_one_box
from utils.general import xywh2xyxy

line_color=(0,0,255)
line_thickness=3
im_w = 720
im_h = 960

root = "/Users/samuelbuckner/Documents/College_Graduate/Coursework/CSE455_ComputerVision/CSE455_Final/yolov5/"
img_path  = root + "tests/swarmed/fuse/"
lab_path  = root + "tests/swarmed/fuse/labels/"
save_path = root + "tests/swarmed/fuse_anno/"

img_list = glob.glob(img_path + "*.png")
lab_list = []

for img in img_list:
    head, tail = os.path.split(img)
    lab_list.append((lab_path + tail).replace(".png", ".txt"))

N = 100
i = 1
for img, lab in zip(img_list, lab_list):

    image  = cv2.imread(img, cv2.IMREAD_UNCHANGED)

    try:
        labels = np.loadtxt(lab, ndmin=2)
    except:
        labels = []
        # head, tail = os.path.split(img)
        # cv2.imwrite(save_path + tail, image)
        # continue

    four_channel_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    four_channel_image.shape

    r = four_channel_image[:, :, 0]
    g = four_channel_image[:, :, 1]
    b = four_channel_image[:, :, 2]
    d = four_channel_image[:, :, 3]

    img1 = np.dstack((r,g,b))
    img2 = np.dstack((d,d,d))
    annotator1 = Annotator(img1, line_width=line_thickness)
    annotator2 = Annotator(img2, line_width=line_thickness)

    for label in labels:
        xywh = label[1:5]
        xywh[0] *= im_w
        xywh[1] *= im_h
        xywh[2] *= im_w
        xywh[3] *= im_h

        xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4))).view(-1).tolist()
        label_txt = "Puck {:.2f}".format(label[5])

        annotator1.box_label(xyxy, label_txt, color=line_color)
        annotator2.box_label(xyxy, label_txt, color=line_color)

    box1 = annotator1.result()
    box2 = annotator2.result()
    box  = np.concatenate((box1, box2), axis=1)

    head, tail = os.path.split(img)
    cv2.imwrite(save_path + tail, box)

    i += 1
    if i % N == 0:
        print("{:n} images processed...".format(i))

    # cv2.namedWindow('annotations', cv2.WINDOW_NORMAL)
    # cv2.imshow('Annotated', box)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()