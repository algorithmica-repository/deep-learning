from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
import utils
import cats_dogs_data_preparation as prep
import pandas as pd
import os

img_width, img_height = 150, 150
epochs = 50
batch_size = 20
top_model_weights_path = 'C:\\Users\\Thimma Reddy\\bottleneck_features\\bottlenck_model.h5'


train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    prep.preapare_small_dataset_for_flow(
                            train_dir_original='C:\\Users\\data\\train', 
                            test_dir_original='C:\\Users\\data\\test',
                            target_base_dir='C:\\Users\\Thimma Reddy\\data1')

base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))


top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))

top_model.load_weights(top_model_weights_path)

model = Model(inputs = base_model.input, outputs = top_model(base_model.output))


set_trainable = False
for layer in model.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
#for layer in model.layers[:15]:
#    layer.trainable = False
    
print(model.summary())


model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])

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

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
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

historydf = pd.DataFrame(history.history, index=history.epoch)

utils.plot_loss_accuracy(history)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
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