from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import utils

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, \
                           n_informative=2, random_state=0, n_clusters_per_class=1)
print(X.shape)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=100)
print(X_train.shape)
print(X_validation.shape)

utils.plot_data(X_train, y_train)

#perceptron model for binary classification
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, verbose=3, epochs=1000, validation_data=(X_validation,y_validation), batch_size=10)
print(model.summary())
print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)

utils.plot_loss_accuracy(history)

y_pred = model.predict_classes(X, verbose=0)

