from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf.plot(ylim=(0, max(1, historydf.values.max())))
    loss = history.history['loss'][-1]
    acc = history.history['acc'][-1]
    plt.title('Loss: %.3f, Accuracy: %.3f' % (loss, acc))


X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, \
                           n_informative=2, random_state=0, n_clusters_per_class=1)
print(X.shape)

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.33, random_state=100)
print(X_train.shape)
print(X_validation.shape)

plot_data(X_train, y_train)

#perceptron model for binary classification
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, verbose=3, epochs=1000, validation_data=(X_validation,y_validation), batch_size=10)
print(model.summary())
print(model.get_weights())

historydf = pd.DataFrame(history.history, index=history.epoch)

plot_loss_accuracy(history)

y_pred = model.predict_classes(X, verbose=0)

