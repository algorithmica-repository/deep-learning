import sys
sys.path.append("G:/New Folder/utils")
from keras.layers import Dense, Input
from keras import Model
import keras_utils as kutils
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import numpy as np
import os

path = "G:/3.hand written digit recognizer(kaggle)"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X_train = digit_train.iloc[:,1:]/255.0
y_train = digit_train['label']

y_train = np_utils.to_categorical(y_train)
print(X_train.shape)
print(y_train1.shape)

input = Input(shape=(784,))
hidden1 = Dense(20, activation='relu')(input)
hidden2 = Dense(20, activation='relu')(hidden1)
output = Dense(10, activation='softmax')(hidden2)

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

X_test = digit_test/255.0
X_test['Label'] = np.argmax(model.predict(X_test), axis=1)
X_test['ImageId'] = list(range(1,X_test.shape[0]+1))
X_test.to_csv(os.path.join(path, "submission.csv"), index=False, columns=['ImageId', 'Label'])