from keras.preprocessing.image import ImageDataGenerator
import PIL

data_dir = 'D:\\sample'
save_dir = 'D:\\preview'

datagen = ImageDataGenerator(rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=3,
    class_mode='binary', save_to_dir=save_dir)

#maps classes to integers based on lexicographic names of sub-directories
print(generator.class_indices)

#reads the next batch of data
data_batch, labels_batch = next(generator)
print('data batch shape:', data_batch.shape)
print('labels batch shape:', labels_batch.shape)

