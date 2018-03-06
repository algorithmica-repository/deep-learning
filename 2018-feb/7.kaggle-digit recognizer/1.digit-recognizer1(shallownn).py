from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os
import utils
from keras.callbacks import ModelCheckpoint, EarlyStopping

os.chdir("F:/")
np.random.seed(100)

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()

X_train = digit_train.iloc[:,1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])

model = Sequential()
model.add(Dense(10,  input_shape=(784,), activation='softmax'))

print(model.summary())
for layer in model.layers:
    print(layer.name, layer.input.shape, layer.output.shape)


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 2
batchsize = 32

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(x=X_train, y=y_train, epochs=epochs, 
                    batch_size=batchsize, validation_split=0.2,
                    callbacks=[early_stopping, save_weights])
print(model.get_weights())
utils.plot_loss_accuracy(history)

digit_test = pd.read_csv("test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test/255.0
pred = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)), "Label": pred})
submissions.to_csv("submission.csv", index=False, header=True)