from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Conv2D, MaxPooling2D, Flatten, RepeatVector, GRU, Bidirectional, TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import pandas as pd

label_length = 5
n_classes = 10
n_train_samples = 100
n_test_samples = 100
epochs = 10
batch_size = 32

def make_dataset(X, y, num_samples, length):
    height = 28
    width = 28*length
    samples = np.ndarray(shape=(num_samples, width, height), dtype=np.float32)
    labels = []
    permutation = np.random.permutation(X.shape[0])
    
    start = 0
    for i in range(num_samples):
        rand_indices = [permutation[index] for index in range(start, start + length)]
        sample = np.hstack([X[index] for index in rand_indices])
        label = [y[index] for index in rand_indices]
        start += length 
        if start >= len(permutation):
                permutation = np.random.permutation(X.shape[0])
                start = 0
        samples[i,:,:] = sample.T
        labels.append(label)
    return {"images": samples, "labels": np.array(labels)}

def show_img_label(train_data, i):
    img = train_data.get('images')[i]
    plt.imshow(img.T, cmap='gray')
    label = train_data.get('labels')[i]
    print(label)
    plt.show()

def prepare_images(img_data):   
    return np.expand_dims(img_data, -1).astype('float32')/255.0

(digits_images_train, digits_labels_train), (digits_images_test, digits_labels_test) = mnist.load_data()
train_data = make_dataset(digits_images_train, digits_labels_train, n_train_samples, label_length)
show_img_label(train_data, 10)
X_train = prepare_images(train_data.get('images'))
y_train = train_data.get('labels')
y_train = np.array([np_utils.to_categorical(i, num_classes=n_classes) for i in y_train])

#convolutional & max pool layers to extract features out of image    
input_data = Input(shape=X_train.shape[1:], name='the_input')
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(input_data)
inner = MaxPooling2D((2, 2), name='max1')(inner)
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(inner)
inner = MaxPooling2D((2, 2), name='max2')(inner)

feature_vector = Flatten()(inner)
inner = Dense(1024, activation='relu', name='dense1')(feature_vector)
inner = RepeatVector(label_length)(inner)
inner = Bidirectional(GRU(64, return_sequences=True, name='gru'))(inner)
output = TimeDistributed(Dense(n_classes, activation='softmax', name='dense2'))(inner)

model = Model(inputs = input_data, outputs = output)
print(model.summary())
for layer in model.layers:
    print(layer.name, layer.input.shape, layer.output.shape)

model.compile(Adam(lr=0.01), 'categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
model.fit(x=X_train, y=y_train, epochs=epochs, 
                    batch_size=batch_size, validation_split=0.1,
                    callbacks=[save_weights])

test_data = make_dataset(digits_images_test, digits_labels_test, n_test_samples, label_length)
show_img_label(test_data, 1)
X_test = prepare_images(test_data.get('images'))
y_test = test_data.get('labels')

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=-1)

def make_str(a):
    res = []
    for y in a:
        res.append(','.join('%d' %x for x in y))
    return res
y_test_str = make_str(y_test)
y_pred_str = make_str(y_pred)
df = pd.DataFrame({'y_test':y_test_str, 'y_pred':y_pred_str})
print(df)