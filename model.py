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

#Train data
train_zip = ZipFile(train_path, 'r')

#Define names and labels
train_df = data.set_index('id')
train_names = train_df.index.values
train_labels = np.asarray(train_df['label'].values)

#Split train data to use 90% for training and 10% for validation. 
X_train, X_valid, y_train, y_valid = train_test_split(train_names, train_labels, test_size=0.1, random_state=42)


#Create onehot encoding for labels
y_train = to_categorical(y_train, num_classes=2)
y_valid = to_categorical(y_valid, num_classes=2)

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 200
data_augmentation = True #Check exactly what kind of augmentation is performed
						 #Rotation and differential cropping ma be useful for making the
						 #network robust
num_classes = 2


#Parameters
train_size = 60000
test_size = 10000


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

#Get train and valid data
img_size = 96
crop_size = 48
start_crop = (img_size - crop_size)//2
end_crop = start_crop + crop_size


def images_to_arrays(names, directory):
	'''A function that fetches all the images, converts them to numpy arrays
	and returns them as one whole numpy array
	'''
	data_list = []
	for name in names:
		#Get image as array
		img_array = get_data(name+'.tif', train_zip)
		#Crop image
		#A positive label is called if the center 32x32 pixels have at least one cancer polyp.
		#Cropping surely reduces the problem, but additional information from outside the crop zone is lost
		#All of this has to be taken into consideration. The crop size can thus be varied.


		img_array = img_array[start_crop:end_crop, start_crop:end_crop]
		data_list.append(img_array)

	return np.array(data_list)

X_train = images_to_arrays(X_train, train_zip)
X_valid = images_to_arrays(X_valid, train_zip)


depth = 20
# Input image dimensions.
input_shape = X_train.shape[1:]
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# lr scheduler
#lr_scheduler = LearningRateScheduler(lr_schedule) #Reduces learning rate during training to avoid jumping out of optimal minima




model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=1,
              validation_data=(X_valid, y_valid),
              shuffle=True)


# Score trained model.
#scores = model.evaluate(x_test, y_test, verbose=1)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])


#Save model to disk
from tensorflow.keras.models import model_from_json   
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")