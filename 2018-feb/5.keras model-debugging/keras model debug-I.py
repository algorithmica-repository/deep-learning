from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
import utils

os.chdir("F:/")
np.random.seed(100)

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()

X_train = digit_train.iloc[:,1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='sigmoid' ))
model.add(Dense(10,  activation='softmax'))

#get the details of layers and output dimensions
print(model.summary())

#get the details of input and output dimensions
for layer in model.layers:
    print(layer.name, layer.input.shape, layer.output.shape)


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batchsize = 32

history = model.fit(x=X_train, y=y_train, epochs=epochs, batch_size=batchsize, validation_split=0.2)
print(model.get_weights())
utils.plot_loss_accuracy(history)