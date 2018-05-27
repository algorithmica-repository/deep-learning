from keras.layers import Input, Dense, LeakyReLU, Activation
from keras.models import Model
from keras.datasets import mnist 
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import math
import keras

print(keras.__version__)
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
   x =  Dense(128)(input)
   x = LeakyReLU(alpha=0.01)(x)
   x = Dense(784)(x)
   x = Activation('tanh')(x)
   generator = Model(input, x)
   return generator

def test_generator(generator, n_samples, sample_size):
    latent_samples = make_latent_samples(n_samples, sample_size)
    images = generator.predict(latent_samples)
    plot_images(images)

def build_descriminator(descriminator_input_size):
   input = Input(shape=(descriminator_input_size,))     
   x =  Dense(128)(input)
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
    descriminator =  build_descriminator(descriminator_input_size)
    print(descriminator.summary())

    #build adversarial model = generator + discriminator
    gan = Model(input, descriminator(generator(input)))
    print(gan.summary())

    descriminator.compile(optimizer=Adam(lr=d_learning_rate), loss='binary_crossentropy')
    gan.compile(optimizer=Adam(lr=g_learning_rate), loss='binary_crossentropy')
    
    return generator, descriminator, gan


def make_latent_samples(n_samples, sample_size):
    return np.random.uniform(-1, 1, size=(n_samples, sample_size))
    #return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))

def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])

def train_gan(generator, descriminator, gan, X_train_real, X_val_real, batch_size, epochs, latent_sample_size):
    val_size = len(X_val_real)
    y_train_real, y_train_fake = make_labels(batch_size)
    y_val_real,  y_val_fake  = make_labels(val_size)

    
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
            descriminator.trainable = True
            descriminator.train_on_batch(X_batch_real, y_train_real)
            descriminator.train_on_batch(X_batch_fake, y_train_fake)

            # train the generator via GAN
            descriminator.trainable = False
            gan.train_on_batch(latent_samples, y_train_real)
    
        # evaluate at end of epoch
        latent_samples = make_latent_samples(val_size, latent_sample_size)
        X_val_fake = generator.predict_on_batch(latent_samples)

        d_loss  = descriminator.test_on_batch(X_val_real, y_val_real)
        d_loss += descriminator.test_on_batch(X_val_fake, y_val_fake)
        g_loss  = gan.test_on_batch(latent_samples, y_val_real) 
    
        losses.append((d_loss, g_loss))
    
        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
                e+1, epochs, d_loss, g_loss))
    return losses

def plot_loss(losses):
   losses = np.array(losses)
   fig, ax = plt.subplots()
   plt.plot(losses.T[0], label='Discriminator')
   plt.plot(losses.T[1], label='Generator')
   plt.title("Test Losses")
   plt.legend()
   plt.show() 

def preprocess(x):    
    x = x.reshape(-1, 784) # 784=28*28
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

val_percent = 0.2
batch_size = 32
epochs = 20
latent_sample_size = 100
g_learning_rate = 0.0001 
d_learning_rate = 0.001
descriminator_input_size = 784

(X_train, _), (X_test, _) = mnist.load_data()
X_train = preprocess(X_train)
X_test = preprocess(X_test)

#build generator, descriminator and gan
generator, descriminator, gan = build_gan(latent_sample_size, descriminator_input_size, g_learning_rate, d_learning_rate)
#train and validate gan
losses = train_gan(generator, descriminator, gan, X_train, X_test, batch_size, epochs, latent_sample_size)
#plot losses
plot_loss(losses)

#generate images using trained model
test_generator(generator, 16, latent_sample_size) 
 