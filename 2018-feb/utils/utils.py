import os
import shutil
import matplotlib.pyplot as plt
import random

def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        for c in ['dogs', 'cats']: 
            os.mkdir(os.path.join(train_dir, c))
            os.mkdir(os.path.join(validation_dir, c))
        os.mkdir(os.path.join(test_dir, 'images'))
        print('created the required directory structure')
        
        files = os.listdir(train_dir_original)
        train_files = [os.path.join(train_dir_original, f) for f in files]
        random.shuffle(train_files)    
        n = int(len(train_files) * val_percent)
        val = train_files[:n]
        train = train_files[n:]  

        for t in train:
            if 'cat' in t:
                shutil.copy2(t, os.path.join(train_dir, 'cats'))
            else:
                shutil.copy2(t, os.path.join(train_dir, 'dogs'))
     
        for v in val:
            if 'cat' in v:
                shutil.copy2(v, os.path.join(validation_dir, 'cats'))
            else:
                shutil.copy2(v, os.path.join(validation_dir, 'dogs'))
        files = os.listdir(test_dir_original)
        test_files = [os.path.join(test_dir_original, f) for f in files]
        for t in test_files:
            shutil.copy2(t, os.path.join(test_dir, 'images'))
    else:
        print('required directory structure already exists. learning continues with existing data')

    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in ['dogs', 'cats']:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in ['dogs', 'cats']:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    nb_test_samples = len(os.listdir(os.path.join(test_dir, 'images')))
    print('total test images:', nb_test_samples )
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples

def preapare_small_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        for c in ['dogs', 'cats']: 
            os.mkdir(os.path.join(train_dir, c))
            os.mkdir(os.path.join(validation_dir, c))
        os.mkdir(os.path.join(test_dir, 'images'))
        print('created the required directory structure')        
       
        train_cats = ['cat.{}.jpg'.format(i) for i in range(1000)]
        for t in train_cats:
             shutil.copy2(os.path.join(train_dir_original, t), os.path.join(train_dir, 'cats'))
        train_dogs = ['dog.{}.jpg'.format(i) for i in range(1000)]
        for t in train_dogs:
             shutil.copy2(os.path.join(train_dir_original, t), os.path.join(train_dir, 'dogs'))        
        val_cats = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
        for t in val_cats:
             shutil.copy2(os.path.join(train_dir_original, t), os.path.join(validation_dir, 'cats'))
        val_dogs = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
        for t in val_dogs:
             shutil.copy2(os.path.join(train_dir_original, t), os.path.join(validation_dir, 'dogs'))

        files = os.listdir(test_dir_original)           
        test_files = [os.path.join(test_dir_original, f) for f in files]
        for t in test_files:
            shutil.copy2(t, os.path.join(test_dir, 'images'))
    else:
        print('required directory structure already exists. learning continues with existing data')
    
    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in ['dogs', 'cats']:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in ['dogs', 'cats']:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    nb_test_samples = len(os.listdir(os.path.join(test_dir, 'images')))
    print('total test images:', nb_test_samples )

    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples
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
