import numpy as np
import glob

from pathlib import Path
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def create_model(max_frames):
    numClasses=20
    model=Sequential()
    model.add(InputLayer(input_shape=(max_frames, 1900, 1)))
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

in_files = '/home/will/avp_cw2/testing_features/visual_dct_features/8x8/*.mat'
max_frames = 96

data = []
labels = []
for feature_file in sorted(glob.glob(in_files)):
    stemFilename = (Path(os.path.basename(feature_file)).stem)
    label = stemFilename.split('_')
    labels.append(label[0])

# for d in data:
#    max_frames = max(max_frames, d.shape[0])
   
# print(max_frames)   

data = []
for feature_file in sorted(glob.glob(in_files)):
    # feature_data = np.load(feature_file)  
    feature_data = scipy.io.loadmat(feature_file)['videoFeature']
    all_zero_cols = np.all(feature_data==0, axis=0)
    feature_data = feature_data[:, ~all_zero_cols]
    feature_data = np.pad(feature_data, ((0, max_frames - feature_data.shape[0]), (0, 0)) )
    # print(feature_data)
    # print(feature_data.shape)#each data matrix think about the spectrum matrix is being added a couple of zeros in rows and columns where (0 - backwards/down row, 0 - forwards/up), (0 - left, max_frames - mfcc_data.shape[1] - right column)  
    data.append(feature_data) #append the new data matrix into this array with 2 dimensions

labels = np.array(labels)
data = np.array(data)
# print(data.shape)
data = data / np.max(data)


LE=LabelEncoder()
classes = ['Amelia', 'Ben', 'Christopher', 'Danny', 'Emilija', 'Joey', 'Josh', 
           'Kacper', 'Kaleb', 'Konark', 'Krish', 'Leo', 'Louis', 'Muneeb', 
           'Naima', 'Noah', 'Ryan', 'Seb', 'Sebastian', 'Zachary']
LE = LE.fit(classes)
labels=to_categorical(LE.transform(labels))

# Load previously-created network
model = create_model(max_frames)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=0.001))
model.load_weights('/home/will/avp_cw2/models/visual_model/weights/names-8x8.weights.h5')

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