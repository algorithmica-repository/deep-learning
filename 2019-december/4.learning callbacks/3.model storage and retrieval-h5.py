#tensorboard --logdir=foo:G:/logs
import sys
sys.path.append("I:/New Folder/utils")
import regression_utils as rutils
from keras.layers import Dense
from keras import Sequential, models
import keras_utils as kutils
from keras import metrics
from sklearn import model_selection
from keras.callbacks import TensorBoard, EarlyStopping
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

path="I:/"
early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')   
tensorboard = TensorBoard(log_dir=os.path.join(path, "logs"),  histogram_freq=1,
                              write_images=True)

history = model.fit(x=X_train, y=y_train, verbose=3, epochs=100,  batch_size=16, validation_split=0.1,
                    callbacks=[early_stopping, tensorboard] )
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
rutils.plot_data_2d_regression(X_train, y_train)

#serialization to h5
model.save(os.path.join(path, "model.h5"))

#deserialization from h5
loaded_model = models.load_model(os.path.join(path, "model.h5"))
print(loaded_model.summary())
print(loaded_model.get_weights())

y_pred = loaded_model.predict(X_test)
rutils.regression_performance(model, X_test, y_test)
