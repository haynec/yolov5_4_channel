import numpy as np
import cv2 as cv
import scipy.io
from PIL import Image
import time
import os
import errno
import random
from sklearn.model_selection import train_test_split
import shutil
import cv2
###############################################################################


###############################################################################
# Reads _dep and _vis images and stacks them into 4-dimensional image;
# Also divides dataset into vis_only, dep_only, and fused_only, and partitions
# data into train and test folders using 70-30 split
# note: for some reason, the train and test folders end up inside the "/all/" 
# directory for vis and dep, but not for fused...I'll trouble shoot this later,
# but for now this script generates the folders needed for training and we can 
# just move them to the outter directory (i.e., "vis_only")
###############################################################################
fuse_data = False

dep_dir = '/home/chris/cse455_final/Annotated/dep'
vis_dir = '/home/chris/cse455_final/Annotated/vis'
fused_dir = '/home/chris/cse455_final/Annotated/fuse'

directories = [dep_dir, vis_dir, fused_dir]
image_types = ['dep', 'vis', 'fused']
dep_annotations = [os.path.join(dep_dir, x) for x in os.listdir(dep_dir) if x[-3:] == "txt"]
dep_images = [os.path.join(dep_dir, x) for x in os.listdir(dep_dir) if x[-3:] == "jpg"]
vis_annotations = [os.path.join(vis_dir, x) for x in os.listdir(vis_dir) if x[-3:] == "txt"]
vis_images = [os.path.join(vis_dir, x) for x in os.listdir(vis_dir) if x[-3:] == "jpg"]

dep_annotations.sort()
dep_images.sort()
vis_annotations.sort()
vis_images.sort()




if fuse_data:
    for index, image in enumerate(dep_images):
        dep_data = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        vis_data = cv2.imread(vis_images[index], cv2.IMREAD_UNCHANGED)
        fused_img = np.dstack((vis_data,dep_data))
        old_path, fname = os.path.split(image)
        name, ext = os.path.splitext(fname)
        new_name = name.replace('dep', 'fused.png')
        cv2.imwrite(os.path.join(fused_dir, new_name), fused_img)
#        print(vis_data.shape)
#        print(dep_data.shape)
#        print(fused_img.shape)
#        cv2.imshow('dep', dep_data)
#        cv2.imshow('vis', vis_data)
#        cv2.imshow('fused', fused_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
else: 
    fused_images = [os.path.join(fused_dir, x) for x in os.listdir(fused_dir) if x[-3:] == "png"]
    fused_annotations = [os.path.join(fused_dir, x) for x in os.listdir(fused_dir) if x[-3:] == "txt"]
    fused_annotations.sort()
    fused_images.sort()
    
    for k, outter_dir in enumerate(directories):
        #temp = os.path.split(outter_dir)
        #out_dir = os.path.join(temp[0])
        #print(temp[0])
        image_type = image_types[k]
        if image_type == 'dep':
            images_master = dep_images
            annotations_master = dep_annotations
        elif image_type == 'vis':
            images_master =  vis_images
            annotations_master = vis_annotations
        elif image_type == 'fused':
            images_master = fused_images
            annotations_master = fused_annotations
        try:
            print(outter_dir)
            os.makedirs(os.path.join(outter_dir,'test'))
            os.makedirs(os.path.join(outter_dir,'train'))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        if len(os.listdir(os.path.join(outter_dir,'test')))==0:
            # does 70-30 split between train and test (https://realpython.com/train-test-split-python-data/)
            x_train, x_test, y_train, y_test = train_test_split(images_master,annotations_master)
    
            for index,image in enumerate(x_train):
                shutil.copy2(image,os.path.join(outter_dir,'train/'))
                shutil.copy2(y_train[index],os.path.join(outter_dir,'train/'))
    
            for index,image in enumerate(x_test):
                shutil.copy2(image,os.path.join(outter_dir,'test/'))
                shutil.copy2(y_test[index],os.path.join(outter_dir,'test/'))
        else: 
            print('dir is not empty')      