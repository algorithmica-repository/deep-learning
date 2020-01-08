import sys
sys.path.append("J:/New Folder/utils")
from keras.layers import Dense
from keras import Sequential
import keras_utils as kutils
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import os


path = "J:/3.hand written digit recognizer(kaggle)"

digit_train = pd.read_csv(os.path.join(path, "train.csv"))
print(digit_train.shape)
print(digit_train.info())

X_train = digit_train.iloc[:,1:]/255.0
y_train = digit_train['label']

y_train1 = np_utils.to_categorical(y_train)
print(X_train.shape)
print(y_train1.shape)


model = Sequential()
model.add(Dense(units=20, input_shape=(784,), activation='sigmoid'))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint(os.path.join(path, "model.h5"), monitor='val_loss', save_best_only=True)   

history = model.fit(x=X_train, y=y_train1, verbose=3, epochs=100,  batch_size=32, validation_split=0.1,
                callbacks=[early_stopping, save_model] )
print(model.summary())
print(model.get_weights())
print(history.history['accuracy'][-1])
print(history.history['val_accuracy'][-1])
kutils.plot_loss(history)

digit_test = pd.read_csv(os.path.join(path,"test.csv"))
digit_test.shape
digit_test.info()

X_test = digit_test/255.0
pred = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred})
submissions.to_csv(os.path.join(path, "submission.csv"), index=False, header=True)
