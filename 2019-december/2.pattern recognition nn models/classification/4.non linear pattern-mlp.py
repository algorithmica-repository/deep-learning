import sys
sys.path.append("I:/New Folder/utils")
import classification_utils as cutils
import clustering_utils as cl_utils
from keras.layers import Dense
from keras import Sequential
import keras_utils as kutils
from keras.utils import np_utils
from sklearn import model_selection

#2-d classification pattern
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.1)
X, y = cl_utils.generate_synthetic_data_2d_clusters(n_samples=1000, n_centers=4, cluster_std=1.2 )

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

y_train1 = np_utils.to_categorical(y_train)

#single layered perceptron  model
def getModel1():
    model = Sequential()
    model.add(Dense(units=2, input_shape=(2,), activation='softmax'))
    return model

#multiple layer perceptron model(12-2)
def getModel2():
    model = Sequential()
    model.add(Dense(units=100, input_shape=(2,), activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model

#multiple layer perceptron model(12-8-2)   
def getModel3():
    model = Sequential()
    model.add(Dense(units=12, input_shape=(2,), activation='sigmoid'))
    model.add(Dense(units=8, activation='sigmoid'))
    model.add(Dense(units=2, activation='softmax'))
    return model

def getModel4():
    model = Sequential()
    model.add(Dense(units=12, input_shape=(2,), activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))
    return model

def getModel5():
    model = Sequential()
    model.add(Dense(units=12, input_shape=(2,), activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))
    return model

model = getModel3()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train1, verbose=3, epochs=100,  batch_size=10, validation_split=0.1)
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
cutils.plot_model_2d_classification(model, X_train, y_train, use_keras=True)

y_pred = model.predict_classes(X_test)
cutils.performance_metrics_hard_multiclass_classification(model, X_test, y_test, use_keras=True)


history = model.fit(x=X_train, y=y_train1, verbose=3, epochs=100,  batch_size=32, validation_split=0.1)
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
cutils.plot_model_2d_classification(model, X_train, y_train, use_keras=True)

y_pred = model.predict_classes(X_test)
cutils.performance_metrics_hard_multiclass_classification(model, X_test, y_test, use_keras=True)