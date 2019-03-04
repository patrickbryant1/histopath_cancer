#! /usr/bin/env python3
# -*- coding: utf-8 -*-



import pdb
import numpy as np
import pandas as pd
import os
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from PIL import Image
from io import StringIO

#Image information: all images are 96x96 color images with 3 channels (r,g,b)
data = pd.read_csv('/home/patrick/Documents/data/histopath_cancer/train_labels.csv')
train_path = '/home/patrick/Documents/data/histopath_cancer/train.zip'
test_path = '/home/patrick/Documents/data/histopath_cancer/test.zip'

# quick look at the label stats
def get_counts(data):
	print(data['label'].value_counts())

get_counts(data)

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
#train_files = train_zip.infolist()

#Define names and labels
train_df = data.set_index('id')
train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

#Split train data to use 90% for training and 10% for validation. 
tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)
img = get_data(tr_n[0]+'.tif', train_zip)


#A positive label is called if the center 32x32 pixels have at least one cancer polyp.
#Cropping surely reduces the problem, but additional information from outside the crop zone is lost
#All of this has to be taken into consideration. The crop size can thus be varied.