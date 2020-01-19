from keras.preprocessing.image import ImageDataGenerator
import PIL

data_dir = 'D:/cats vs dogs/sample_train'
save_dir = 'D:/save'

#Keras supports ImageDataGenerator allows us to quickly set up 
#Python generators that can automatically turn image files on #disk into batches of pre-processed tensors
datagen = ImageDataGenerator(rescale=1. / 255,
        rotation_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(256, 256),
    batch_size=4,
    class_mode='binary', save_to_dir=save_dir)

#maps classes to integers based on lexicographic names of sub-directories
print(generator.class_indices)

#reads the next batch of data
data_batch, labels_batch = next(generator)
print('data batch shape:', data_batch.shape)
print('labels batch shape:', labels_batch.shape)