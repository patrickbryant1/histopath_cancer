#! /usr/bin/env python3
# -*- coding: utf-8 -*-



import pdb
import numpy as np
import pandas as pd
import os
import random
from sklearn.utils import shuffle
from zipfile import ZipFile
from PIL import Image
from io import StringIO

train_labels = pd.read_csv('/home/patrick/Documents/data/histopath_cancer/train_labels.csv')
train_path = '/home/patrick/Documents/data/histopath_cancer/train.zip'
test_path = '/home/patrick/Documents/data/histopath_cancer/test.zip'
# quick look at the label stats
print(train_labels['label'].value_counts())


def get_data(filename, archive):
	'''A function for getting the .zip .tif images
	into numpy arrays
	'''
	img = Image.open(archive.open(filename))
	img_array = np.array(img)

	return img_array


#Test data
test_zip = ZipFile(test_path, 'r')
test_files = test_zip.infolist()

#Train data
train_zip = ZipFile(train_path, 'r')
train_files = train_zip.infolist()

#for i in range(0, len(train_files)):
get_data(train_files[0].filename, train_zip)
pdb.set_trace()
