from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
import utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

os.chdir("F:/")
np.random.seed(100)

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()

X = digit_train.iloc[:,1:].values.astype('float32')
X_images=X.reshape(X.shape[0],28,28,1)
y = np_utils.to_categorical(digit_train["label"])

X_images_train, X_images_validation, y_train, y_validation = train_test_split(X_images, y, stratify=y, test_size=0.20, random_state=100)
print(X_images_train.shape)
print(X_images_validation.shape)

epochs = 4
batch_size = 32
img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(10,  activation='softmax'))

print(model.summary())
for layer in model.layers:
    print(layer.name, layer.input.shape, layer.output.shape)

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1. / 255)

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_datagen.flow(X_images_train, y_train, batch_size=batch_size),
    steps_per_epoch=X_images_train.shape[0]//batch_size,
    epochs=epochs,
    validation_data= validation_datagen.flow(X_images_validation, y_validation, batch_size=batch_size),
    validation_steps=X_images_validation.shape[0]//batch_size,
    callbacks=[save_weights])

print(model.get_weights())
utils.plot_loss_accuracy(history)

digit_test = pd.read_csv("test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0
X_test_images=X_test.reshape(X_test.shape[0],28,28,1)

pred = model.predict_classes(X_test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred})
submissions.to_csv("submission.csv", index=False, header=True)