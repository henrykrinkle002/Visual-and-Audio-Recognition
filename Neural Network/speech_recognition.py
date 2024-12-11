#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 15:00:35 2024

@author: amalkurian
"""

import numpy as np
import glob

from pathlib import Path
import os

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Flatten, Conv2D, InputLayer, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from sklearn import metrics

in_files = '/home/will/avp_cw2/training_features/audio_mfcc_features/*.npy'

data = []
labels = []
for mfcc_file in sorted(glob.glob(in_files)):
    mfcc_data = np.load(mfcc_file)
    data.append(mfcc_data)
    stemFilename = (Path(os.path.basename(mfcc_file)).stem)
    label = stemFilename.split('_')
    labels.append(label[0])
    
print(len(data[0]))    

print(data[0].shape)
print(data[1].shape)

max_frames = 214
# for d in data:
#    max_frames = max(max_frames, d.shape[0])
   
print(max_frames)   

data = []
for mfcc_file in sorted(glob.glob(in_files)):
    mfcc_data = np.load(mfcc_file)  
    mfcc_data = np.pad(mfcc_data, ((0, max_frames - mfcc_data.shape[0]), (0, 0)) )
    #print(mfcc_data.shape)#each data matrix think about the spectrum matrix is being added a couple of zeros in rows and columns where (0 - backwards/down row, 0 - forwards/up), (0 - left, max_frames - mfcc_data.shape[1] - right column)  
    data.append(mfcc_data) #append the new data matrix into this array with 2 dimensions

labels = np.array(labels)
data = np.array(data)
print(data.shape)
data = data / np.max(data)


LE=LabelEncoder()
classes = ['Amelia', 'Ben', 'Christopher', 'Danny', 'Emilija', 'Joey', 'Josh', 
           'Kacper', 'Kaleb', 'Konark', 'Krish', 'Leo', 'Louis', 'Muneeb', 
           'Naima', 'Noah', 'Ryan', 'Seb', 'Sebastian', 'Zachary']
LE = LE.fit(classes)
labels=to_categorical(LE.transform(labels))


# X_test, X_tmp, y_test, y_tmp = train_test_split(data, labels, test_size=0.9, random_state=0, stratify=labels)
X_val, X_train, y_val, y_train = train_test_split(data, labels, test_size=0.8, random_state=0)

# Only need training and validation data, test data has been provided separately
# X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.20, random_state=0)

## Initial model definition.
# def create_model():
#     numClasses=20
#     model=Sequential()
#     model.add(InputLayer(input_shape=(164, 10, 1)))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3)))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(numClasses))
#     model.add(Activation('softmax'))
#     return model

def create_model():
    numClasses=20
    model=Sequential()
    model.add(InputLayer(input_shape=(214, 10, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(numClasses))
    model.add(Activation('softmax'))
    return model

model = create_model()
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], 
              optimizer=Adam(learning_rate=0.001))
model.summary()

num_batch_size = 16
num_epochs = 50


# Train neural network with num_epochs
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=num_batch_size, epochs=num_epochs, verbose=1)
# Get accuracy and add to list
# predicted_probs = model.predict(X_test, verbose=0) #predict probabilities based on test dataset, don't print progress as verbosity = 0
# predicted = np.argmax(predicted_probs, axis=1) #predicted_probs is a testdataset converts the probablities into predicted class labels.
# actual = np.argmax(y_test, axis=1) #same but Y testdata and actual class labels
# accuracy = metrics.accuracy_score(actual, predicted)   #differnce


model.save_weights('/home/will/avp_cw2/models/audio_model/weights/names.weights.h5')


# Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# predicted_probs = model.predict(X_test, verbose=0) #predict probabilities based on test dataset, don't print progress as verbosity = 0
# predicted = np.argmax(predicted_probs, axis=1) #predicted_probs is a testdataset converts the probablities into predicted class labels.
# actual = np.argmax(y_test, axis=1) #same but Y testdata and actual class labels
# accuracy = metrics.accuracy_score(actual, predicted)   #differnce
# print(f'Accuracy: {accuracy * 100}%')
# predicted_classes = LE.inverse_transform(predicted)
# actual_classes = LE.inverse_transform(actual)

# cm_display = metrics.ConfusionMatrixDisplay.from_predictions(actual_classes, predicted_classes, labels=classes, xticks_rotation='vertical')
# cm_display.plot()
# plt.margins(50)
# plt.show()