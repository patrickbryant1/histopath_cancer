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
from tensorflow.keras.callbacks import TensorBoard
import time
import math

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
                  default=sys.stdin, help = '''path to atore log files for tensorboard.''')

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





def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4)) #L2 regularization with penalty 10exp-4

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x



def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes: (if 32x32 images and 16 filters)
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes 

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=2)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


#################
#MAIN
#################

#Parameters
#Crop parameters
img_size = 96 #Don't change
crop_size = 48
start_crop = (img_size - crop_size)//2
end_crop = start_crop + crop_size

# Training parameters
batch_size = 128  # orig paper trained all networks with batch_size=128
num_epochs = 20
num_classes = 2
max_lr = 0.01
min_lr = max_lr/10
lr_change = (max_lr-min_lr)/5 #Reduce further lst three epochs
#Opt
find_lr = False

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


X_train = images_to_arrays(X_train, train_zip)
X_train = X_train#/255 #rescaling by 255 (make pixel intensities into 0 to 1 range)
X_valid = images_to_arrays(X_valid, train_zip)
X_valid = X_valid#/255 #Rescale

#Keras datagenerator, performs augmentation
#random rotation by 90 deg
#random horizontal and vertical flips
#rescaling by 255 (make pixel intensities into 0 to 1 range)
datagen = ImageDataGenerator(rotation_range = 90)
                            #horizontal_flip = True) 
                              #vertical_flip = False)  
                              #zoom_range = 32.0)
                              #brightness_range = [0.0,10.0])
                              #zca_whitening = True)

datagen.fit(X_train)


depth = 20 #6n+2, where n is the number of resnet layers
# Input image dimensions.
input_shape = X_train.shape[1:]
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#Write summary of model
model.summary()

#Checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

#One cycle lr optimization function 
save_lrate = []
def one_cycle(epochs):
  '''Increase lr each batch to find start lr
  '''
  step = epochs
  initial_rate = 0.000001
  final_rate = 1
  #steps_per_epoch = len(X_train) / batch_size
  #interval = steps_per_epoch/100
  interval = (math.log(final_rate)-math.log(initial_rate))/100

  lrate = math.log(initial_rate)+(interval*step)
  print(lrate)
  save_lrate.append(lrate)
  lrate = math.exp(lrate)

  return lrate

def lr_schedule(epochs):
  '''lr scheduel according to one-cycle policy.
  '''
  
  #Increase lrate in beginning
  if epochs == 0:
    lrate = min_lr
  elif (epochs <6 and epochs > 0):
    lrate = min_lr+(epochs*lr_change)
  #Decrease further below min_lr last three epochs
  elif epochs > 10:
    lrate = min_lr/(10*(epochs-10))
  #After the max lrate is reached, decrease it back to the min
  else:
    lrate = max_lr-((epochs-5)*lr_change)

  print(epochs,lrate)
  return lrate

if find_lr == True:
  lrate = LearningRateScheduler(one_cycle)
  callbacks = [lrate]
  steps_per_epoch = (len(X_train) / batch_size)/100
  num_epochs = 100
  validation_data=(X_valid[0:100], y_valid[0:100])
else:
  lrate = LearningRateScheduler(lr_schedule)
  callbacks=[tensorboard, checkpoint, lrate]
  steps_per_epoch = (len(X_train) / batch_size)
  validation_data=(X_valid, y_valid)

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size),
              steps_per_epoch=steps_per_epoch,
              epochs=num_epochs,
              validation_data=validation_data,
              shuffle=True, #Dont feed continuously
              callbacks=callbacks) #, lr_scheduler])

#Print lr and loss
if find_lr == True:
  losses = history.history['loss']
  for i in range(0,len(save_lrate)):
    print(save_lrate[i], losses[i])

if find_lr == False:
#Save model to disk
#from tensorflow.keras.models import model_from_json   
#serialize model to JSON
  model_json = model.to_json()
  with open("./models/model."+log_name+".json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
  model.save_weights("./models/model."+log_name+".h5")
  print("Saved model to disk")
