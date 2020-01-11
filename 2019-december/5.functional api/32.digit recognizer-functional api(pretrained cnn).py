import sys
sys.path.append("G:/New Folder/utils")
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import Model
import keras_utils as kutils
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import os
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions

path = "G:/3.hand written digit recognizer(kaggle)"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X_train = digit_train.iloc[:,1:].values.astype('float32')/255.0
X_train = X_train.reshape(X_train.shape[0],28,28,1)
y_train = np_utils.to_categorical(digit_train["label"])

img_width, img_height = 32, 32

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

vgg_model = VGG16(include_top=False, input_shape=input_shape)
print(type(vgg_model))
print(vgg_model.summary())
for layer in vgg_model.layers:
    print(layer.name, layer.input, layer.output)
print(vgg_model.input)
print(vgg_model.output)

features = Flatten()(vgg_model.get_layer('block4_pool').output)
hidden1 = Dense(30, activation='relu')(features)
output = Dense(10, activation='softmax')(hidden1)

model = Model(inputs=vgg_model.input, outputs=output)
print(model.summary())