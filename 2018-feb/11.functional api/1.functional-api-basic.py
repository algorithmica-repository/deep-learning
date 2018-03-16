#sequenctial models using functional API
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras import Model

#input sample size of 2 , 1 output perceptron
input = Input(shape=(2,))
output = Dense(1)(input)
model1 = Model(inputs=input, outputs=output)
print(model1.summary())

input = Input(shape=(2,))
tmp = Dense(1)
output = tmp(input)
model1 = Model(inputs=input, outputs=output)
print(model1.summary())

#input sample size of 10, 3 hidden layers with 10,20,10 perceptrons, output layer with 1 peceptron
input = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(input)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model2 = Model(inputs=input, outputs=output)
print(model2.summary())

#input image of size 64*64*1, 2 conv-pool layers of 32 and 16, 1 hidden layer of size 10, output layer of size 2
input = Input(shape=(64,64,1))
conv1 = Conv2D(32, kernel_size=4, activation='relu')(input)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
hidden1 = Dense(10, activation='relu')(pool2)
output = Dense(2, activation='softmax')(hidden1)
model3 = Model(inputs=input, outputs=output)
print(model3.summary())