#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:00:58 2022

@author: chris
"""

import cv2
import numpy as np

line_thickness=3

# exactly one box case
img = '/home/chris/four_channel/airsim_fused_dataset/scene_only/val/rp_carla_rigged_001_ue4_4_None_scene.png'
box = '/home/chris/four_channel/airsim_fused_dataset/scene_only/val/rp_carla_rigged_001_ue4_4_None_scene.txt'

# more than one box case
img = '/home/chris/four_channel/airsim_fused_dataset/scene_only/val/rp_dennis_posed_004_Mobile_33_Dust_scene.png'
box = '/home/chris/four_channel/airsim_fused_dataset/scene_only/val/rp_dennis_posed_004_Mobile_33_Dust_scene.txt'

image = cv2.imread(img)
h, w = image.shape[0:2]

boxes = np.loadtxt(box)

# detect empty text files or reshape boxes as needed
if len(boxes.shape)==0 or boxes.size==0:
    print('validate_boxes, input shape 0 -> empty txt file')
elif len(boxes.shape)==1: 
    print("validate_boxes, input shape 1 -> only 1 box, need to reshape")
    boxes = boxes.reshape((1,boxes.shape[0]))
 
num_boxes = boxes.shape[0]

for k in range(0, num_boxes):
    centerX = boxes[k,1]*w
    centerY = boxes[k,2]*h
    box_w   = boxes[k,3]*w
    box_h   = boxes[k,4]*h
    pt1 = (int(centerX - box_w//2), int(centerY - box_h//2))
    pt2 =(int(centerX + box_w//2), int(centerY + box_h//2)) 
    box = cv2.rectangle(image, pt1, pt2, [0,255,0],line_thickness)
cv2.namedWindow('annotations', cv2.WINDOW_NORMAL)
cv2.imshow('annotations', image)
cv2.imshow('annotations', box)
cv2.waitKey(0); 
cv2.destroyAllWindows()