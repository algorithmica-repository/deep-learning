from keras import applications, optimizers
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import Model, Sequential

base_model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(150, 150, 3))


top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(512, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.2))
top_model.add(Dense(2, activation='softmax'))

model = Model(inputs = base_model.input, outputs = top_model(base_model.output))
