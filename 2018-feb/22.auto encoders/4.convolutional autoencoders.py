from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model
from keras.datasets import mnist 
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint

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
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

input_size = 784
epochs = 5
batch_size = 256

input_img = Input(shape=(28, 28, 1)) 
x = Conv2D(8, (3, 3), padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 16)
x = Conv2DTranspose(16, (3, 3), padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
print(autoencoder.summary())

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2, callbacks=[save_weights])

decoded_imgs = autoencoder.predict(X_test)
plot(10, X_test, decoded_imgs)

encoder = Model(input_img, encoded)
tmp = encoder.predict(X_test)
