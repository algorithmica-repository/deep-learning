from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D, Dropout
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

def prepare_labels(labels, n_classes):
    dig0_arr = np.ndarray(shape=(len(labels),n_classes))
    dig1_arr = np.ndarray(shape=(len(labels),n_classes))
    dig2_arr = np.ndarray(shape=(len(labels),n_classes))
    dig3_arr = np.ndarray(shape=(len(labels),n_classes))  
    dig4_arr = np.ndarray(shape=(len(labels),n_classes))    
    for index,label in enumerate(labels):
        dig0_arr[index,:] = np_utils.to_categorical(label[0],n_classes)
        dig1_arr[index,:] = np_utils.to_categorical(label[1],n_classes)
        dig2_arr[index,:] = np_utils.to_categorical(label[2],n_classes)
        dig3_arr[index,:] = np_utils.to_categorical(label[3],n_classes)
        dig4_arr[index,:] = np_utils.to_categorical(label[4],n_classes)        
    return [dig0_arr,dig1_arr,dig2_arr,dig3_arr,dig4_arr]


(digits_images_train, digits_labels_train), (digits_images_test, digits_labels_test) = mnist.load_data()
train_data = make_dataset(digits_images_train, digits_labels_train, n_train_samples, label_length)
show_img_label(train_data, 50)
X_train = prepare_images(train_data.get('images'))
y_train = train_data.get('labels')
y_train = prepare_labels(train_data.get('labels'), n_classes)

#convolutional & max pool layers to extract features out of image    
input_data = Input(shape=X_train.shape[1:], name='the_input')
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv1')(input_data)
inner = MaxPooling2D((2, 2), name='max1')(inner)
inner = Conv2D(16, (3,3), padding='same', activation='relu', kernel_initializer='he_normal', name='conv2')(inner)
inner = MaxPooling2D((2, 2), name='max2')(inner)
features = Flatten()(inner)
inner = Dense(100, activation='relu')(features)
inner = Dropout(0.2)(inner)
inner = Dense(100, activation='relu')(inner)
inner = Dropout(0.2)(inner)
branches = []
for i in range(5):
    branches.append(Dense(n_classes, activation='softmax')(inner))
model = Model(inputs=input_data, outputs=branches)
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

def decode_pred(y_pred):
    predictions=[]
    for i in range(n_test_samples):
        pred = []
        for j in range(5):
            pred.append(np.argmax(y_pred[j][i]))
        predictions.append(pred)
    return np.array(predictions)
y_pred = decode_pred(y_pred)

def make_str(a):
    res = []
    for y in a:
        res.append(','.join('%d' %x for x in y))
    return res
y_test_str = make_str(y_test)
y_pred_str = make_str(y_pred)
df = pd.DataFrame({'y_test':y_test_str, 'y_pred':y_pred_str})
print(df)