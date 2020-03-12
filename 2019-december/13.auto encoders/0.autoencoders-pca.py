from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist 
import matplotlib.pyplot as plt
import sys
sys.path.append("I:/New Folder/utils")
import classification_utils as cutils
from sklearn import manifold

def plot(n, X, Decoded_X):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # original
        plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
 
        # reconstruction
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(Decoded_X[i].reshape(28, 28))
        plt.gray()
        plt.axis('off')
 
    plt.tight_layout()
    plt.show()
 
(X_train, y_train), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((X_train.shape[0], -1))
X_test = X_test.reshape((X_test.shape[0], -1))

input_size = 784
encoding_size = 64
epochs = 10
batch_size = 64

 #linear pca
input = Input(shape=(input_size,))
encoded = Dense(encoding_size, activation='linear')(input)
decoded = Dense(input_size, activation='linear')(encoded)
autoencoder = Model(input, decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

decoded_imgs = autoencoder.predict(X_test)
plot(20, X_test, decoded_imgs)

encoder = Model(input, encoded)
tmp = encoder.predict(X_train)

tsne = manifold.TSNE()
tsne_data = tsne.fit_transform(tmp)
cutils.plot_data_2d_classification(tsne_data, y_train)

 #non-linear pca
input = Input(shape=(input_size,))
encoded = Dense(encoding_size, activation='sigmoid')(input)
decoded = Dense(input_size, activation='linear')(encoded)
autoencoder = Model(input, decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

decoded_imgs = autoencoder.predict(X_test)
plot(20, X_test, decoded_imgs)

encoder = Model(input, encoded)
tmp = encoder.predict(X_train)

tsne = manifold.TSNE()
tsne_data = tsne.fit_transform(tmp)
cutils.plot_data_2d_classification(tsne_data, y_train)
