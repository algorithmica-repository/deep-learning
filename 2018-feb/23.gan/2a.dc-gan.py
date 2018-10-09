from keras.layers import Input, Dense, LeakyReLU, Activation, Reshape, Conv2D, Conv2DTranspose, Activation, BatchNormalization, Flatten
from keras.models import Model
from keras.datasets import mnist 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import math
import keras

def plot_images(images):
    plt.figure(figsize=(20, 8))
    num_images = images.shape[0]
    rows = int(math.sqrt(images.shape[0]))
    for i in range(num_images):
        plt.subplot(rows, rows, i + 1)
        image = deprocess(images[i])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def build_generator(input):
    x = Dense(784)(input)
    x = Reshape(target_shape=(7, 7, 16))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
        
    x = Conv2DTranspose(32, (5,5), strides=2, padding='same')(x) 
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
        
    x = Conv2DTranspose(1, (5,5), strides=2, padding='same')(x)
    x = Activation('tanh')(x)
    generator = Model(input, x)
    return generator

def test_generator(generator, n_samples, sample_size):
    latent_samples = make_latent_samples(n_samples, sample_size)
    images = generator.predict(latent_samples)
    plot_images(images)

def build_descriminator():
    input = Input(shape=(28, 28, 1)) 
    x = Conv2D(32, (5, 5), strides=2, padding='same')(input)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = Conv2D(16, (5, 5), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    descriminator = Model(input, x)
    return descriminator  

def build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate):
    input = Input(shape=(latent_sample_size,))     

    #build generator model
    generator = build_generator(input)
    print(generator.summary())

    #build descriminator model
    descriminator =  build_descriminator()
    print(descriminator.summary())

    #build adversarial model = generator + discriminator
    gan = Model(input, descriminator(generator(input)))
    print(gan.summary())

    descriminator.compile(optimizer=Adam(lr=d_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    return generator, descriminator, gan


def make_latent_samples(n_samples, sample_size):
    return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    #return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])

def train_gan(generator, descriminator, gan, X_train_real, batch_size, epochs, latent_sample_size):
    y_train_real, y_train_fake = make_labels(batch_size)
    
    losses = []
    nbatches = len(X_train_real) // batch_size
    for e in range(epochs):
        for i in range(nbatches):
            # real MNIST digit images
            X_batch_real = X_train_real[i*batch_size:(i+1)*batch_size]
        
            # latent samples and the generated digit images
            latent_samples = make_latent_samples(batch_size, latent_sample_size)
            X_batch_fake = generator.predict_on_batch(latent_samples)
        
            # train the discriminator to detect real and fake images
            X = np.concatenate((X_batch_real, X_batch_fake))
            y = np.concatenate((y_train_real, y_train_fake))
            descriminator.trainable = True
            metrics = descriminator.train_on_batch(X, y)
            d_loss = metrics[0]
            d_acc = metrics[1]
            log = "%d/%d: [discriminator loss: %f, acc: %f]" % (e, i, d_loss, d_acc)

            # train gan with latent_samples and y_train_real of 1s
            descriminator.trainable = False
            metrics = gan.train_on_batch(latent_samples, y_train_real)
            
            g_loss = metrics[0]
            g_acc = metrics[1]
            log = "%s [adversarial loss: %f, acc: %f]" % (log, g_loss, g_acc)
            print(log)
            losses.append((d_loss, g_loss, d_acc, g_acc))

    return losses

def plot_loss(losses):
   losses = np.array(losses)
   fig, ax = plt.subplots()
   plt.plot(losses.T[0], label='Discriminator')
   plt.plot(losses.T[1], label='GAN')
   plt.title("Train Losses")
   plt.legend()
   plt.show() 

def preprocess(x):    
    x = x.reshape(-1, 28, 28, 1)
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x

def deprocess(x):
    x = (x / 2 + 1) * 255
    x = np.clip(x, 0, 255)
    x = np.uint8(x)
    x = x.reshape(28, 28)
    return x       

batch_size = 64
epochs = 2
latent_sample_size = 100
g_learning_rate = 0.0001 
d_learning_rate = 0.001

(X_train, _), (_, _) = mnist.load_data()
X_train = preprocess(X_train)

#build generator, descriminator and gan
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)

#generate images using trained model
test_generator(generator, 25, latent_sample_size) 
 