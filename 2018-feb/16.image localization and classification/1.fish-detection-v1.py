import custom_generator as generator
import bbox_utils as utils
import data_preparation_fishes as prep
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras import applications, optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import keras.backend as K
import numpy as np
import pandas as pd
        
train_path = 'C:/Users/Thimma Reddy/courses-algorithmica/big data deep learning/4.projects-deep learning/fish detection/train'
test_path  = 'C:/Users/Thimma Reddy/courses-algorithmica/big data deep learning/4.projects-deep learning/fish detection/test_stg1'
boxs_path ='C:/Users/Thimma Reddy/courses-algorithmica/big data deep learning/4.projects-deep learning/fish detection/bounding-boxes'

img_width, img_height = 224, 224
epochs = 10
batch_size = 20

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
file2sizes = utils.get_file_sizes(train_path)
file2boxes = utils.get_bounding_boxes(boxs_path)
desired_size = (img_width, img_width)
file2boxes = utils.adjust_bounding_boxes(file2boxes, file2sizes, desired_size)

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    prep.preapare_full_dataset_for_flow(
                            train_dir_original = train_path, 
                            test_dir_original = test_path,
                            target_base_dir = 'C:\\Users\\Thimma Reddy\\data3')

    
base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))

p = 0.2
x = base_model.output
x = MaxPooling2D()(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)

for layer in base_model.layers:
    layer.trainable=False    
model = Model(inputs=base_model.input, outputs=[x_bb, x_class])
print(model.summary())

model.compile(optimizers.Adam(lr=0.001), 
              loss=['mse', 'categorical_crossentropy'], 
              metrics=['accuracy'],
              loss_weights=[.001, 1.])
train_generator = generator.DirectoryIterator(train_dir,  target_size= (img_width, img_height), 
                                            batch_size=batch_size,  shuffle=True, map_extras=file2boxes)

validation_generator = generator.DirectoryIterator(validation_dir,  target_size= (img_width, img_height), 
                                            batch_size=batch_size,  shuffle=True, map_extras=file2boxes)

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights])

test_generator = generator.DirectoryIterator(test_dir,  target_size= (img_width, img_height), 
                                            class_mode = None, batch_size=batch_size,  shuffle=True)
preds = model.predict_generator(test_generator, nb_test_samples//batch_size)

class_probs = utils.do_clip(preds[1],0.82)
df = pd.DataFrame(class_probs, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
img_ids = np.array([f.split('\\')[-1] for f in test_generator.filenames])
df.insert(0, 'image', img_ids)
df.to_csv('submission.csv', index=False)
