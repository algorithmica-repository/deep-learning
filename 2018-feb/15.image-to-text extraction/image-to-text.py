from random import randrange
from matplotlib import patches
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

img_width, img_height = 28, 28
nclasses = 10
epochs = 10
batchsize = 32
n_train_samples = 100

def make_dataset(X, y, num_samples, length, debug=False):
    height = img_height
    width = img_width*length
    samples = np.ndarray(shape=(num_samples, height, width), dtype=np.float32)
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
        if debug:
            print(label)
            plt.imshow(sample, cmap='gray')
            plt.show()
        samples[i,:,:] = sample
        labels.append(label)
    return {"examples": samples, "labels": labels}

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

def prepare_images(img_data):   
    return np.expand_dims(img_data, -1).astype('float32')/255.0

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_data = make_dataset(x_train, y_train, n_train_samples, 5, True)
train_labels = prepare_labels(train_data.get('labels'), nclasses)
train_images = prepare_images(train_data.get('examples'))

input = Input(shape=(28, 140, 1))
conv = Conv2D(32, (3, 3), activation='relu')(inputs)
conv = MaxPooling2D((2, 2))(conv)
conv = Conv2D(64, (3, 3), activation='relu')(conv)
conv = MaxPooling2D((2, 2))(conv)
features = Flatten()(conv)

branches = []
for i in range(5):
    branches.append(Dense(nclasses, activation='softmax')(features))
model = Model(input, branches)
print(model.summary())

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)
history = model.fit(x=train_images, y=train_labels, epochs=epochs, 
                    batch_size=batchsize, validation_split=0.2,
                    callbacks=[save_weights])
print(model.get_weights())
utils.plot_loss_accuracy(history)


n_test_samples = 100
test_data = make_dataset(x_test, y_test, n_test_samples, 5, True)
test_labels = prepare_labels(test_data.get('labels'), nclasses)
test_images = prepare_images(test_data.get('examples'))

predictions = model.predict(test_images)

def calculate_acc(predictions, real_labels, n_test_samples):    
    individual_counter = 0
    global_sequence_counter = 0
    for i in range(0,len(predictions[0])):
        #Reset sequence counter at the start of each image
        sequence_counter = 0 
        
        for j in range(0,5):
            if np.argmax(predictions[j][i]) == np.argmax(real_labels[j][i]):
                individual_counter += 1
                sequence_counter +=1
        
        if sequence_counter == 5:
            global_sequence_counter += 1
         
    ind_accuracy = individual_counter/n_test_samples * 5.0
    global_accuracy = global_sequence_counter/n_test_samples
    
    return ind_accuracy,global_accuracy

calculate_acc(predictions, test_labels, n_test_samples)