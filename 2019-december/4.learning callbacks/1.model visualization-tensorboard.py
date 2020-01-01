#tensorboard --logdir=foo:G:/logs
import sys
sys.path.append("I:/New Folder/utils")
import regression_utils as rutils
from keras.layers import Dense
from keras import Sequential
import keras_utils as kutils
from keras import metrics
from sklearn import model_selection
from keras.callbacks import TensorBoard
import os

X, y = rutils.generate_nonlinear_synthetic_data_regression(n_samples=200, n_features=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

def getModel1():
    model = Sequential()
    model.add(Dense(units=1, input_shape=(1,), activation='linear'))
    return model

def getModel2():
    model = Sequential()
    model.add(Dense(units=10, input_shape=(1,), activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    return model

def getModel3():
    model = Sequential()
    model.add(Dense(units=20, input_shape=(1,), activation='relu'))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    return model

model = getModel3()

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=[metrics.mean_squared_error])
tensorboard = TensorBoard(log_dir=os.path.join("I:/", "logs"),  histogram_freq=1,
                              write_images=True)

history = model.fit(x=X_train, y=y_train, verbose=3, epochs=100,  batch_size=16, validation_split=0.1,
                    callbacks=[tensorboard] )
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
rutils.plot_model_2d_regression(model, X_train, y_train)

y_pred = model.predict_classes(X_test)
cutils.performance_metrics_hard_multiclass_classification(model, X_test, y_test, use_keras=True)