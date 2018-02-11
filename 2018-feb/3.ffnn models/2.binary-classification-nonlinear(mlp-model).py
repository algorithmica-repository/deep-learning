from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary
from sklearn.datasets import make_moons, make_circles
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model

X, y = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
#X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)
plot_data(X, y)

#single perceptron model for binary classifcation
model1 = Sequential()
model1.add(Dense(1, input_shape=(2,), activation='sigmoid'))

model1.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
plot_model(model1, show_shapes=True, to_file='model1.png')

history1 = model1.fit(X, y, verbose=0, epochs=100)
plot_loss_accuracy(history1)
plot_decision_boundary(lambda x: model1.predict(x), X, y)

y_pred = model1.predict_classes(X, verbose=0)
plot_confusion_matrix(model1, X, y)

#mlp model for binary classification
model2 = Sequential()
model2.add(Dense(4, input_shape=(2,), activation='tanh'))
model2.add(Dense(2, activation='tanh'))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(Adam(lr=0.01), 'binary_crossentropy', metrics=['accuracy'])
plot_model(model2, show_shapes=True, to_file='model2.png')

history2 = model2.fit(X, y, verbose=0, epochs=50)

plot_loss_accuracy(history2)
plot_decision_boundary(lambda x: model2.predict(x), X, y)

y_pred = model2.predict_classes(X, verbose=0)
plot_confusion_matrix(model2, X, y)