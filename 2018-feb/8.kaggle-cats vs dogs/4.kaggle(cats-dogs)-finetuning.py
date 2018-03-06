from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import backend as K
import pandas as pd
import utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import VGG16


os.getcwd()
train_dir, validation_dir, test_dir = utils.prepare_data_flow_directory(
                            train_dir_original='C:\\Users\\data\\train', 
                            test_dir_original='C:\\Users\\data\\test',
                            target_base_dir='C:\\Users\\Thimma Reddy\\data1')

img_width, img_height = 150, 150
nb_train_samples = 2000
nb_validation_samples = 1000
nb_test_samples = 12500
epochs = 2
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
    
    
model = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# add the model on top of the convolutional base
model.add(top_model)

#model.trainable = True
for layer in model.layers:
    if layer.name == 'block5_conv1':
        layer.trainable = True
    else:
        layer.trainable = False
    
print(model.summary())


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


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
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')   
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
print(test_generator.filenames)
probabilities = model.predict_generator(test_generator, nb_test_samples//batch_size)

mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = probabilities[i][0]
    i += 1
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})
tmp.to_csv('submission.csv', columns=['id','label'],index=False)