#-------------------------------------------------------------------------------
# Name:        Emotion Recognition Using Facial Expressions with CNNs
# Purpose:
#
# Author:      sivaprasadrb
#
# Created:     10/11/2018
# Copyright:   (c) sivaprasadrb 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from __future__ import print_function
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import optimizers

import numpy as np
import matplotlib.pyplot as plt

data_path = './dataset/'

imgs = np.empty((256, 256), int)

filenames = sorted(os.listdir(data_path))

classificationLabels = []
count = 0
for img_name in filenames:
    img = plt.imread(data_path + img_name)
    img  = np.resize(img, (256, 256))
    if p == 0:
	imgs = (img)
	classificationLabels = 1
    else:
    	imgs = np.append(imgs, img, axis=0)
    classificationLabels.append(int(img_name[1]))
imgs = np.reshape(imgs, [ 445, 256, 256])

train_images, test_images, train_labels, test_labels = train_test_split(imgs, classificationLabels, test_size=0.33, random_state=42)

from keras.utils import to_categorical

print('Training data shape : ', train_images.shape, len(train_labels))
print('Testing data shape : ', test_images.shape, len(test_labels))

classes = np.unique(train_labels)
classes=np.append(classes,0)
nClasses = len(classes)

print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)
plt.figure(figsize=[4,2])

plt.subplot(121)
plt.imshow(train_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_labels[0]))

plt.subplot(122)
plt.imshow(test_images[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_labels[0]))

print(train_images.shape[1:])
nRows,nCols = train_images.shape[1:]
nDims = nRows
print(nCols)
train_data = train_images.reshape(train_images.shape[0], nRows, nCols, 1)
test_data = test_images.reshape(test_images.shape[0], nRows, nCols, 1)
input_shape = (nRows, nCols, 1)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data /= 255
test_data /= 255

print(len(train_labels))
print(len(test_labels))

train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)