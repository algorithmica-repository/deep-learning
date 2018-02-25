import os
import shutil
import matplotlib.pyplot as plt

def copy_images(fnames, src, target):
    for fname in fnames:
        shutil.copyfile(os.path.join(src, fname), os.path.join(target, fname))

def prepare_data_flow_directory(train_dir_original, test_dir_original, target_base_dir):
    os.mkdir(target_base_dir)

    # Directories for our training, validation and test split
    train_dir = os.path.join(target_base_dir, 'train')
    os.mkdir(train_dir)
    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)

    validation_dir = os.path.join(target_base_dir, 'validation')
    os.mkdir(validation_dir)
    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)

    test_dir = os.path.join(target_base_dir, 'test')
    os.mkdir(test_dir)
    # Directory with our test pictures
    test_images_dir = os.path.join(test_dir, 'images')
    os.mkdir(test_images_dir)    
        
    # Copy first 1000 cat images to train_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    copy_images(fnames, train_dir_original, train_cats_dir)
    # Copy first 1000 dog images to train_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    copy_images(fnames, train_dir_original, train_dogs_dir)
 
    # Copy next 500 cat images to validation_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    copy_images(fnames, train_dir_original, validation_cats_dir)
    # Copy next 500 dog images to validation_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    copy_images(fnames, train_dir_original, validation_dogs_dir)

    # Copy all images to test__dir
    fnames = ['{}.jpg'.format(i) for i in range(1, 12501)]
    copy_images(fnames, test_dir_original, test_images_dir)

    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test images:', len(os.listdir(test_images_dir)))
    
    return train_dir, validation_dir, test_dir

def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()


