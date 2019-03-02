#! /usr/bin/env python3
# -*- coding: utf-8 -*-



import pdb
import numpy as np
import pandas as pd
import os
import cv2
import random
from sklearn.utils import shuffle

data = pd.read_csv('/home/patrick/Documents/data/train_labels.csv')
#train_path = '/kaggle/input/train/'
#test_path = '/kaggle/input/test/'
# quick look at the label stats
#data['label'].value_counts()
pdb.set_trace()

def readImage(path):
    # OpenCV reads the image in bgr format by default
    bgr_img = cv2.imread(path)
    # We flip it to rgb for visualization purposes
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return rgb_img

