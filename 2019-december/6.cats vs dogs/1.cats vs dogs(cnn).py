#dropout and data-augmentation to reduce overfitting
import sys
sys.path.append("I:/New Folder/utils")
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
#import collections
import os
import pandas as pd
import keras_utils as kutils
import data_preparation as prep
from keras.callbacks import ModelCheckpoint, EarlyStopping

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    prep.preapare_small_dataset_for_flow(
                            train_dir_original='D:\\cats vs dogs\\train', 
                            test_dir_original='D:\\cats vs dogs\\test',
                            target_base_dir='D:\\small_data')

img_width, img_height = 256, 256
epochs = 1
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
input = Input(shape=input_shape)
conv1 = Conv2D(32, (3, 3), activation='relu') (input)
pool1 = MaxPooling2D((2, 2)) (conv1)
conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
pool2 = MaxPooling2D((2, 2)) (conv2)
features =  Flatten() (pool2)
fc1 = Dense(512, activation='relu') (features)
dropout1 = Dropout(0.5)(fc1)
output = Dense(2, activation='softmax') (dropout1)

model = Model(inputs = input, outputs=output)
print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

train_image_generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
train_generator = train_image_generator.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_image_generator = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_image_generator.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, early_stopping])

kutils.plot_loss_accuracy(history)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

tmp = next(test_generator)
act = kutils.get_activations(model, tmp[0:1])  # with just one sample.
kutils.display_activations(act, directory=os.path.join("D:/cats vs dogs", 'digit_activations'), save=True)
kutils.display_heatmaps(act,tmp[0:1], directory=os.path.join("D:/cats vs dogs", 'digit_heatmaps'), save=True)

#print(test_generator.filenames)
probabilities = model.predict_generator(test_generator, nb_test_samples//batch_size)

mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][1]
    i += 1
#od = collections.OrderedDict(sorted(mapper.items()))    
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission.csv', columns=['id','label'], index=False)