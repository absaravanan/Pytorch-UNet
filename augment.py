import cv2
import os
import imutils
from random import randint
import random

root = "/media/bubbles/fecf5b15-5a64-477b-8192-f8508a986ffe/ai/abs/aadhaar_mask/train/"
target_dir = "/media/bubbles/fecf5b15-5a64-477b-8192-f8508a986ffe/ai/abs/aadhaar_aug_mask/"

for r, d, f in os.walk(root+"images"):
    for file in f:
        print (file)
        filename = os.path.join(r, file)
        anno_filename = os.path.join(root+"annotations/",file)
        # print (anno_filename)
        onlyfilename, file_extension = os.path.splitext(file)
        if os.path.isfile(anno_filename):
            angles = random.sample(range(0, 360), 10)
            for angle in angles:
                img = cv2.imread(filename)
                annotation = cv2.imread(anno_filename)
                # angle = randint(0,360)

                rot_img = imutils.rotate_bound(img, angle)
                rot_annotation = imutils.rotate_bound(annotation, angle)

                rot_img_name = target_dir + "images/" + onlyfilename+"_"+str(angle)+file_extension
                rot_annotation_name = target_dir + "annotations/"+ onlyfilename+"_"+str(angle)+file_extension
                cv2.imwrite(rot_img_name, rot_img)
                cv2.imwrite(rot_annotation_name, rot_annotation)


