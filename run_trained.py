#! /usr/bin/env python3
# -*- coding: utf-8 -*-



import pdb
import numpy as np
import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
from PIL import Image
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical   
import time
import tensorflow

from tensorflow.keras.models import model_from_json 
import argparse
import sys


#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
 
parser.add_argument('json_file', nargs=1, type= str,
                  default=sys.stdin, help = 'path to .json file with keras model to be opened')

parser.add_argument('weights', nargs=1, type= str,
                  default=sys.stdin, help = '''path to .h5 file containing weights for net.''')

parser.add_argument('data_path', nargs=1, type= str,
                  default=sys.stdin, help = '''path to data to be used for prediction.''')


def load_model(json_file, weights):

    global model

    json_file = open(json_file, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights)
    model._make_predict_function()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_data(filename, archive):
	'''A function for getting the .zip .tif images
	into numpy arrays
	'''
	img = Image.open(archive.open(filename))
	img_array = np.array(img)

	return img_array


def images_to_arrays(names, directory):
	'''A function that fetches all the images, converts them to numpy arrays
	and returns them as one whole numpy array
	'''
	data_list = []
	for name in names:
		#Get image as array
		img_array = get_data(name, directory)
		#Crop image
		#A positive label is called if the center 32x32 pixels have at least one cancer polyp.
		#Cropping surely reduces the problem, but additional information from outside the crop zone is lost
		#All of this has to be taken into consideration. The crop size can thus be varied.


		img_array = img_array[start_crop:end_crop, start_crop:end_crop]
		data_list.append(img_array)

	return np.array(data_list)

def write_pred(names, called):
	'''Write the called predictions for each id to a file in the format:
	id, label
	0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
	95596b92e5066c5c52466c90b69ff089b39f2737,0
	248e6738860e2ebcf6258cdc1f32f299e0c76914,0
	etc.
	'''
	if len(called) != len(names):
		raise ValueError('The number of predictions does not equal the number of ids')

	print('id, label')
	for i in range(0,len(names)):
		name = names[i].split('.tif')[0]
		print(name+','+str(called[i]))

	return None

#Crop parameters
img_size = 96
crop_size = 48
start_crop = (img_size - crop_size)//2
end_crop = start_crop + crop_size
#Main program
args = parser.parse_args()

#Inputs
json_file = (args.json_file[0])
weights = (args.weights[0])
data_path = (args.data_path[0])

#Get files
test_zip = ZipFile(data_path, 'r')
#get names
names = test_zip.namelist()
#Get numpy array
x_test = images_to_arrays(names, test_zip)

#Load and run model
model = load_model(json_file, weights)
pred = model.predict(x_test)

argmax_pred = tensorflow.argmax(pred, 1)

sess = tensorflow.Session()
called = sess.run(argmax_pred)

#Write predictions to stdout
write_pred(names, called)
