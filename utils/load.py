#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image
from os import path
import cv2
import random

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    ids = []
    for f in os.listdir(dir):
        print (f)
        # if path.isfile("/home/ai/ai/data/coco/annotations/"+f):
        ids.append(f[:-4])
    return tuple(ids)
        # return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale, angle):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # print (dir + id + suffix)       
        # img = Image.open(dir + id + suffix)
        # w, h = img.size
        # ima = Image.new('RGB', (w,h))
        # ima = np.transpose(ima, axes=[2, 0, 1])
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale, angle=angle)
        # cv2.imshow("1", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        yield im

def to_cropped_masks(ids, dir, suffix, scale, angle):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        # print (dir + id + suffix)       
        img = Image.open(dir + id + suffix)
        # w, h = img.size
        # ima = Image.new('RGB', (w,h))
        # ima = np.transpose(ima, axes=[2, 0, 1])
        gray = img.convert('L')
        bw = gray.point(lambda x: 0 if x<128 else 255, '1')
        im = resize_and_crop(bw, scale=scale, angle=angle)
        # cv2.imshow("1", im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        yield im


def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    angle = random.randint(0,360)

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale, angle)
    

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_masks(ids, dir_mask, '.jpg', scale, angle)
#     masks_normalized = map(normalize, masks)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '.jpg')
    return np.array(im), np.array(mask)
