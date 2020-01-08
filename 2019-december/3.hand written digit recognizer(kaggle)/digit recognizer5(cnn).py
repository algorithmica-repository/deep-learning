import sys
sys.path.append("J:/New Folder/utils")
from keras import Sequential
import keras_utils as kutils
from keras.utils import np_utils
import pandas as pd
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

path = "J:/3.hand written digit recognizer(kaggle)"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X_train = digit_train.iloc[:,1:].values.astype('float32')/255.0
X_train_images=X_train.reshape(X_train.shape[0],28,28,1)
y_train = np_utils.to_categorical(digit_train["label"])

img_width, img_height = 28, 28

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_width, img_height)
else:
    input_shape = (img_width, img_height, 1)

model = Sequential()
model.add(Conv2D(8, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(4, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10,  activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint(os.path.join(path, "model.h5"), monitor='val_loss', save_best_only=True)   

history = model.fit(x=X_train_images, y=y_train, verbose=2, epochs=3,  batch_size=32, validation_split=0.1,
                    callbacks=[early_stopping, save_model] )
print(model.summary())
print(model.get_weights())
print(history.history['acc'][-1])
print(history.history['val_acc'][-1])
kutils.plot_loss(history)

digit_test = pd.read_csv(os.path.join(path,"test.csv"))
digit_test.shape
digit_test.info()

X_test = digit_test.values.astype('float32')/255.0
X_test_images=X_test.reshape(X_test.shape[0],28,28,1)
pred = model.predict_classes(X_test_images)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred})
submissions.to_csv(os.path.join(path, "submission2.csv"), index=False, header=True)

act = kutils.get_activations(model, X_test_images[2:3])  # with just one sample.
kutils.display_activations(act, directory=os.path.join(path, 'digit_activations'), save=True)
kutils.display_heatmaps(act, X_test_images[2:3], directory=os.path.join(path, 'digit_heatmaps'), save=True)
