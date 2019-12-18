import sys
sys.path.append("E:/New Folder/utils")

import classification_utils1 as cutils
from keras.layers import Dense
from keras import Sequential
import keras_utils as kutils
from keras.utils import np_utils

from sklearn import model_selection

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=3, weights=[0.3,0.3,0.4], class_sep=1.5)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

y_train1 = np_utils.to_categorical(y_train)

model = Sequential()
model.add(Dense(units=3, input_shape=(2,), activation='sigmoid'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train1, verbose=3, epochs=100,  batch_size=10, validation_split=0.1)
print(model.summary())
print(model.get_weights())
kutils.plot_loss(history)
cutils.plot_model_2d_classification(model, X_train, y_train)


y_pred = model.predict_classes(X_test)
kutils.performance_metrics_hard_binary_classification(model, X_test, y_test)
