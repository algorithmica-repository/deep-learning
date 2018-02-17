from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.utils import np_utils
import os

def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()


def plot_loss_accuracy(history):
    plt.figure()
    epochs = range(len(history.epoch))
    plt.plot(epochs, history.history['acc'], 'r', linewidth=3.0)
    plt.plot(epochs,history.history['val_acc'], 'b', linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    
    plt.figure()
    plt.plot(epochs, history.history['loss'], 'r', linewidth=3.0)
    plt.plot(epochs,history.history['val_loss'], 'b', linewidth=3.0)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.legend(['Training Loss', 'Validation Loss'],fontsize=18)
    
    plt.show()



os.chdir("E:/")
np.random.seed(100)

digit_train = pd.read_csv("train.csv")
digit_train.shape
digit_train.info()

X_train = digit_train.iloc[:,1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])

model = Sequential()
model.add(Dense(512, input_shape=(784,), activation='sigmoid' ))
model.add(Dense(10,  activation='softmax'))
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 30
batchsize = 16
history = model.fit(x=X_train, y=y_train, verbose=2, epochs=epochs, batch_size=batchsize, validation_split=0.2)
print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)
plot_loss_accuracy(history)

digit_test = pd.read_csv("test.csv")
digit_test.shape
digit_test.info()

X_test = digit_test/255.0
pred = model.predict_classes(X_test)
submissions=pd.DataFrame({"ImageId": list(range(1,len(pred)+1)),
                         "Label": pred})
submissions.to_csv("submission.csv", index=False, header=True)
