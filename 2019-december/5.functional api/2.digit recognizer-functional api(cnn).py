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

path = "G:/3.hand written digit recognizer(kaggle)"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X_train = digit_train.iloc[:,1:].values.astype('float32')/255.0
X_train = X_train.reshape(X_train.shape[0],28,28,1)
y_train = np_utils.to_categorical(digit_train["label"])

img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

input = Input(shape=input_shape)
conv1 = Conv2D(32, kernel_size=3, activation='relu')(input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=3, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
features = Flatten()(pool2)
hidden1 = Dense(30, activation='relu')(features)
output = Dense(10, activation='softmax')(hidden1)

model = Model(inputs=input, outputs=output)
print(model.summary())
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint(os.path.join(path, "model.h5"), monitor='val_loss', save_best_only=True)   

history = model.fit(x=X_train, y=y_train, verbose=2, epochs=3,  batch_size=32, validation_split=0.1,
                callbacks=[early_stopping, save_model] )

digit_test = pd.read_csv(os.path.join(path,"test.csv"))
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0
X_test = X_test.reshape(X_test.shape[0],28,28,1)

digit_test['Label'] = np.argmax(model.predict(X_test), axis=1)
digit_test['ImageId']  = list(range(1,X_test.shape[0]+1))
digit_test.to_csv(os.path.join(path, "submission.csv"), index=False, columns=['ImageId', 'Label'])

index = digit_test[digit_test.Label == 5]
print(index.head())
act = kutils.get_activations(model, X_test[23:24])
kutils.display_activations(act, directory=os.path.join(path, 'digit_activations'), save=True)
kutils.display_heatmaps(act, X_test_images[0:1], directory=os.path.join(path, 'digit_heatmaps'), save=True)