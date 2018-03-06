from keras.preprocessing.image import ImageDataGenerator
import PIL

data_dir = 'D:\\sample'
save_dir = 'D:\\preview'

datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=3,
    class_mode=None, shuffle=False, save_to_dir=save_dir)

#maps classes to integers based on lexicographic names of sub-directories
print(generator.class_indices)

#no labels generated when we provide class_mode = None
data_batch, labels_batch = next(generator)
data_batch = next(generator)
print('data batch shape:', data_batch.shape)

#returns the names of the files in lexicographic order when #shuffle=False
print(generator.filenames)
