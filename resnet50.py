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
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.utils import to_categorical   
from tensorflow.keras.callbacks import TensorBoard
import time

import argparse
import sys


#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
 
parser.add_argument('train_labels', nargs=1, type= str,
                  default=sys.stdin, help = 'path to train_labels.csv file.')

parser.add_argument('train_path', nargs=1, type= str,
                  default=sys.stdin, help = '''path to train.zip.''')

parser.add_argument('test_path', nargs=1, type= str,
                  default=sys.stdin, help = '''path to test.zip.''')

parser.add_argument('log_path', nargs=1, type= str,
                  default=sys.stdin, help = '''path to store log files for tensorboard.''')


args = parser.parse_args()



#Inputs
#Image information: all images are 96x96 color images with 3 channels (r,g,b)
data = pd.read_csv(args.train_labels[0])
train_path = args.train_path[0] #'/home/pbryant/data/histopath_cancer/train.zip'
test_path = args.test_path[0] #'/home/pbryant/data/histopath_cancer/test.zip'
log_path = args.log_path[0] #'/home/pbryant/Documents/histopath_cancer/logs/'
# quick look at the label stats
print(data['label'].value_counts())

#Tensorboard for logging and visualization
log_name = str(time.time())
tensorboard = TensorBoard(log_dir=log_path+log_name)

def get_data(filename, archive):
	'''A function for getting the .zip .tif images
	into numpy arrays
	'''
	img = Image.open(archive.open(filename))
	img_array = np.array(img)

	return img_array


#Test data
test_zip = ZipFile(test_path, 'r')

#Train data
train_zip = ZipFile(train_path, 'r')

#Define names and labels
train_df = data.set_index('id')
train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

#Split train data to use 90% for training and 10% for validation. 
X_train, X_valid, y_train, y_valid = train_test_split(train_names, train_labels, test_size=0.1, random_state=42)

def split_stats(name, y):
	'''print the split stats
	'''
	print(name+':')
	unique, counts = np.unique(y, return_counts=True)
	print(dict(zip(unique, counts)))
	print(counts[0]/(counts[0]+counts[1])) #print frequency of 0


split_stats('train', y_train)
split_stats('valid', y_valid)

#Create onehot encoding for labels
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

# Training parameters
batch_size = 64  # orig paper trained all networks with batch_size=128
epochs = 10
num_classes = 2


#################
#MAIN
#################


#Crop parameters
img_size = 96 #Don't change
crop_size = 48
start_crop = (img_size - crop_size)//2
end_crop = start_crop + crop_size

#Get train and valid data

def images_to_arrays(names, directory):
    '''A function that fetches all the images, converts them to numpy arrays
    and returns them as one whole numpy array
    '''

    data_list = []
    for name in names:
            #Get image as array
            img_array = get_data(name+'.tif', directory)
            #Crop image
	    #A positive label is called if the center 32x32 pixels have at least one cancer polyp.
	    #Cropping surely reduces the problem, but additional information from outside the crop zone is lost
	    #All of this has to be taken into consideration. The crop size can thus be varied.
            img_array = img_array[start_crop:end_crop, start_crop:end_crop]
            data_list.append(img_array)

    return np.array(data_list)

X_train = images_to_arrays(X_train[0:100], train_zip)
X_train = X_train/255 #rescaling by 255 (make pixel intensities into 0 to 1 range)
X_valid = images_to_arrays(X_valid[0:100], train_zip)
X_valid = X_valid/255 #Rescale

#Keras datagenerator, performs augmentation
#random rotation by 90 deg
#random horizontal and vertical flips
#rescaling by 255 (make pixel intensities into 0 to 1 range)
datagen = ImageDataGenerator(rotation_range = 90, horizontal_flip = True) 
                              #vertical_flip = False)  
                              #zoom_range = 32.0)
                              #brightness_range = [0.0,10.0])
                              #zca_whitening = True)

datagen.fit(X_train)



# Input image dimensions.
input_shape = X_train.shape[1:]

#Load ResNet50 pretrained model
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(crop_size,crop_size,3))#, pooling='avg')#Stores models and weightsd somewhere after loading first time

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Write summary of model
model.summary()


#Checkpoint
filepath="./models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')



# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(X_train, y_train[0:100], batch_size = batch_size),
              steps_per_epoch=len(X_train) / batch_size,
              epochs=epochs,
              validation_data=(X_valid, y_valid[0:100]),
              shuffle=True, #Dont feed continuously
              callbacks=[tensorboard, checkpoint]) #, lr_scheduler])



#Save model to disk
from tensorflow.keras.models import model_from_json   
#serialize model to JSON
model_json = model.to_json()
with open("./models/resnet50."+log_name+".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
#model.save_weights("./models/model."+log_name+".h5")
print("Saved model to disk")
