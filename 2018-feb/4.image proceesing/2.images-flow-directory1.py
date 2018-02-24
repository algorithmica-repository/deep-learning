from keras.preprocessing.image import ImageDataGenerator
import PIL

data_dir = 'D:\\sample'
save_dir = 'D:\\preview'

#Read the picture files.
#Decode the JPEG content to RBG grids of pixels.
#Convert these into floating point tensors.
#Rescale the pixel values (between 0 and 255) to the [0, 1] 

#Keras supports ImageDataGenerator allows us to quickly set up #Python generators that can automatically turn image files on #disk into batches of pre-processed tensors
datagen = ImageDataGenerator(rescale=1. / 255)
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
