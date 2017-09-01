from utilities import plot_confusion_matrix, plot_loss_accuracy, plot_multiclass_decision_boundary, make_multiclass
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical

X, y = make_multiclass(K=3)

#single layered perceptron model for multi-class classfication
model1 = Sequential()
model1.add(Dense(3, input_shape=(2,), activation='softmax'))

model1.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
plot_model(model1, show_shapes=True, to_file='model1.png')

y_cat = to_categorical(y)
history1 = model1.fit(X, y_cat, verbose=0, epochs=20)

plot_loss_accuracy(history1)
plot_multiclass_decision_boundary(model1, X, y)

y_pred = model1.predict_classes(X, verbose=0)
plot_confusion_matrix(model1, X, y)


#mlp model for multi-class classification
model2 = Sequential()
model2.add(Dense(128, input_shape=(2,), activation='tanh'))
model2.add(Dense(64, activation='tanh'))
model2.add(Dense(32, activation='tanh'))
model2.add(Dense(16, activation='tanh'))
model2.add(Dense(3, activation='softmax'))

model2.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
plot_model(model2, show_shapes=True, to_file='model2.png')

y_cat = to_categorical(y)
history2 = model2.fit(X, y_cat, verbose=0, epochs=50)
plot_loss_accuracy(history2)
plot_multiclass_decision_boundary(model2, X, y)

y_pred = model2.predict_classes(X, verbose=0)
plot_confusion_matrix(model2, X, y)