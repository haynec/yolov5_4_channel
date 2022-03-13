from distutils import dep_util
import cv2
import numpy as np
line_thickness=3
# exactly one box case
img = '/home/chris/fused_puck_tests/crash/fuse/crash_fused_0001.png'
box = '/home/chris/yolov5_4_channel/runs/detect/exp21/labels/crash_fused_0001.txt'# more than one box case
image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
print(image.shape)
four_channel_image = cv2.imread(img, cv2.IMREAD_UNCHANGED)
four_channel_image.shape
r = four_channel_image[:, :, 0]
g = four_channel_image[:, :, 1]
b = four_channel_image[:, :, 2]
d = four_channel_image[:, :, 3]
img1 = np.dstack((r,g,b))
img2 = np.dstack((d,d,d))
print(img1.shape)
print(img2.shape)
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
#box = cv2.rectangle(image, pt1, pt2, [0,255,0],line_thickness)
box1 = cv2.rectangle(img1, pt1, pt2, [0,255,0],line_thickness)
box2 = cv2.rectangle(img2, pt1, pt2, [0,255,0],line_thickness)
cv2.namedWindow('annotations', cv2.WINDOW_NORMAL)
#cv2.imshow('annotations', image)
cv2.imshow('Visual', box1)
cv2.imshow('Depth', box2)
cv2.waitKey(0);
cv2.destroyAllWindows()
