import matplotlib.pyplot as plt   
import sys
sys.path.append("I:/New Folder/utils")
import classification_utils as cutils
from keras.layers import Dense
from keras import Sequential
import keras_utils as kutils
from keras.utils import np_utils
from sklearn import model_selection

def getModel1():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), activation='sigmoid'))
    return model

def getModel2():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), activation='softmax'))
    return model
    
#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5,0.5], class_sep=1.5)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

y_train1 = np_utils.to_categorical(y_train)

model = getModel2()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train1, verbose=3, epochs=100,  batch_size=10, validation_split=0.1)
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
cutils.plot_model_2d_classification(model, X_train, y_train, use_keras=True)

y_pred = model.predict_classes(X_test)
cutils.performance_metrics_hard_binary_classification(model, X_test, y_test, use_keras=True)