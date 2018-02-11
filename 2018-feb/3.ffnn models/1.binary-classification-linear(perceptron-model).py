from utilities import plot_data, plot_confusion_matrix, plot_loss_accuracy, plot_decision_boundary
from sklearn.datasets import make_classification
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model


X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=7, n_clusters_per_class=1)
plot_data(X, y)

#perceptron model for binary classification
model = Sequential()
model.add(Dense(units=1, input_shape=(2,), activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x=X, y=y, verbose=0, epochs=50)
plot_model(model, show_shapes=True, to_file='model.png')

plot_loss_accuracy(history)
plot_decision_boundary(lambda x: model.predict(x), X, y)

y_pred = model.predict_classes(X, verbose=0)
plot_confusion_matrix(model, X, y)