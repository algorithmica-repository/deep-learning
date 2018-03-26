import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils
import cats_dogs_data_preparation as prep
import pandas as pd
import os
from keras.utils import np_utils

img_width, img_height = 150, 150
epochs = 50
batch_size = 20

def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, test_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)

        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, nb_train_samples // batch_size)
        np.save(open('bottleneck_features_train.npy', 'wb'),
               bottleneck_features_train)
 
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, nb_validation_samples // batch_size)
        np.save(open('bottleneck_features_validation.npy', 'wb'),
                bottleneck_features_validation)
    
        test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_test = model.predict_generator(
                test_generator, nb_test_samples // batch_size)
        np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
    else:
        print('bottleneck directory already exists')
    
    
train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    prep.preapare_small_dataset_for_flow(
                            train_dir_original='C:\\Users\\data\\train', 
                            test_dir_original='C:\\Users\\data\\test',
                            target_base_dir='C:\\Users\\Thimma Reddy\\data1')

model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))

bottleneck_dir = 'C:\\Users\\Thimma Reddy\\bottleneck_features'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, validation_dir, test_dir, bottleneck_dir)

X_train = np.load(open('bottleneck_features_train.npy','rb'))
train_labels = np.array( 
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))
y_train = np_utils.to_categorical(train_labels)


X_validation = np.load(open('bottleneck_features_validation.npy','rb'))
validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))
y_validation = np_utils.to_categorical(validation_labels)

model = Sequential()
model.add(Flatten(input_shape=X_train.shape[1:]))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('bottlenck_model.h5', monitor='val_loss', save_best_only=True)

history = model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, early_stopping])

utils.plot_loss_accuracy(history)

test_data = np.load(open('bottleneck_features_test.npy','rb'))
probabilities = model.predict_proba(test_data)

test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)
mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = round(probabilities[i][1],3)
    i += 1   
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission.csv', columns=['id','label'], index=False)