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

import scipy.io
from scipy.interpolate import CubicSpline

def create_model(max_frames):
    numClasses=20
    model=Sequential()
    model.add(InputLayer(input_shape=(max_frames, 490, 1)))
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

def visual_feature_interp(visual_feat, audio_feat):
    """
    Return visual features matching the number of frames of the supplied audio
    feature. The time dimension must be the first dim of each feature
    matrix; uses the cubic spline interpolation method - adapted from Matlab.

    Args:
        visual_feat: the input visual features size: (visual_num_frames, visual_feature_len)
        audio_feat: the input audio features size: (audio_num_frames, audio_feature_len)

    Returns:
        visual_feat_interp: visual features that match the number of frames of the audio feature
    """

    audio_timesteps = audio_feat.shape[0]

    # Initialize an array to store interpolated visual features
    visual_feat_interp = np.zeros((audio_timesteps, visual_feat.shape[1]))

    for feature_dim in range(visual_feat.shape[1]):
        cubicSpline = CubicSpline(np.arange(visual_feat.shape[0]), visual_feat[:, feature_dim])
        visual_feat_interp[:, feature_dim] = cubicSpline(np.linspace(0, visual_feat.shape[0] - 1, audio_timesteps))

    return visual_feat_interp

dct_in_dir = '/home/will/avp_cw2/testing_features/visual_dct_features/'
mfcc_in_dir = '/home/will/avp_cw2/testing_features/audio_mfcc_features/30db/'

dct_in_files = '/home/will/avp_cw2/testing_features/visual_dct_features/*.mat'
mfcc_in_files = '/home/will/avp_cw2/testing_features/audio_mfcc_features/30db/*.npy'

labels = []

# Get labels to use
for dct_file in sorted(glob.glob(dct_in_files)):
    stemFilename = (Path(os.path.basename(dct_file)).stem)
    label = stemFilename.split('_')
    labels.append(label[0])

max_frames = 214

video_data = []
audio_data = []
    
# Iterate through 
for mfcc_file in sorted(glob.glob(mfcc_in_files)):
    # Get name of the corresponding DCT feature .mat file
    stemFilename = (Path(os.path.basename(mfcc_file)).stem)
    name_components = stemFilename.split('_')
    dct_file = dct_in_dir + '_'.join((name_components[0], name_components[1], "dataset.mat"))
    
    # Load DCT and MFCC data
    dct_data = scipy.io.loadmat(dct_file)['videoFeature']
    mfcc_data = np.load(mfcc_file)
    
    # Interpolate DCT data to line up with MFCC data
    dct_data = visual_feature_interp(dct_data, mfcc_data)
    
    # Pad data to match maximum frames
    dct_data = np.pad(dct_data, ((0, max_frames - dct_data.shape[0]), (0, 0)) )
    mfcc_data = np.pad(mfcc_data, ((0, max_frames - mfcc_data.shape[0]), (0, 0)) )
    
    video_data.append(dct_data)
    audio_data.append(mfcc_data) #append the new data matrix into this array with 2 dimensions

labels = np.array(labels)

video_data = np.array(video_data)
audio_data = np.array(audio_data)

print(video_data.shape)
print(audio_data.shape)

video_data = video_data / np.max(video_data)
audio_data = audio_data / np.max(audio_data)

data = np.concatenate((video_data, audio_data), axis=2)


LE=LabelEncoder()
classes = ['Amelia', 'Ben', 'Christopher', 'Danny', 'Emilija', 'Joey', 'Josh', 
           'Kacper', 'Kaleb', 'Konark', 'Krish', 'Leo', 'Louis', 'Muneeb', 
           'Naima', 'Noah', 'Ryan', 'Seb', 'Sebastian', 'Zachary']
LE = LE.fit(classes)
labels=to_categorical(LE.transform(labels))

# Load previously-created network
model = create_model(max_frames)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
model.load_weights('/home/will/avp_cw2/models/audio_visual_model/weights/names.weights.h5')

predicted_probs = model.predict(data, verbose=0) #predict probabilities based on test dataset, don't print progress as verbosity = 0
predicted = np.argmax(predicted_probs, axis=1) #predicted_probs is a testdataset converts the probablities into predicted class labels.
actual = np.argmax(labels, axis=1) #same but Y testdata and actual class labels
accuracy = metrics.accuracy_score(actual, predicted)   #differnce
print(f'Accuracy: {accuracy * 100}%')
predicted_classes = LE.inverse_transform(predicted)
actual_classes = LE.inverse_transform(actual)

cm_display = metrics.ConfusionMatrixDisplay.from_predictions(actual_classes, predicted_classes, labels=classes, xticks_rotation='vertical')
# cm_display.plot()
plt.margins(50)
plt.show()