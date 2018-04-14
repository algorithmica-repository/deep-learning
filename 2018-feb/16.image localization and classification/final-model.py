import custom_generator as generator
import bbox_utils as utils

size=(224, 224)
batch_size=10
path = 'C:\\Users\\Thimma Reddy\\courses-algorithmica\\big data deep learning\\4.projects-deep learning\\fish detection\\train'
bpath ='C:\\Users\\Thimma Reddy\\courses-algorithmica\\big data deep learning\\4.projects-deep learning\\fish detection\\bounding-boxes'

batches = generator.DirectoryIterator(directory=path,  target_size= size, bbox_directory = bpath, batch_size=batch_size,  shuffle=True)
tmp = next(batches)
utils.show_bb(tmp,4)



from vgg16bn import Vgg16BN
vgg = Vgg16BN(size,include_top=False)
model = vgg.model

for layer in model.layers:
    layer.trainable=False
p = 0.2
x = model.output
x = MaxPooling2D()(x)
x = BatchNormalization(axis=1)(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)


model = Model(input=model.input, output=[x_bb, x_class])
model.compile(Adam(lr=0.001), 
              loss=['mse', 'categorical_crossentropy'], 
              metrics=['accuracy'],
              loss_weights=[.001, 1.])


model.fit_generator(batches, samples_per_epoch=batches.nb_sample, nb_epoch=epochs)