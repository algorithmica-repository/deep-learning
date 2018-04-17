import os
import shutil
import random

def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')
    classes = []
    for subdir in os.listdir(train_dir_original):
        classes.append(subdir)
    print(classes)

    if os.path.exists(target_base_dir):
        print('required directory structure already exists. learning continues with existing data')
    else:          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        
        for c in classes: 
            os.mkdir(os.path.join(train_dir, c))
            os.mkdir(os.path.join(validation_dir, c))
        print('created the required directory structure')
        
        shutil.move(test_dir_original, test_dir)
        print('moving of test data to target test directory finished')
        
        for c in classes:
            sudir = os.path.join(train_dir_original, c)
            files = os.listdir(sudir)
            train_files = [os.path.join(sudir, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  

            for t in train:
                shutil.copy2(t, os.path.join(train_dir, c))
            for v in val:
                shutil.copy2(v, os.path.join(validation_dir, c))
        print('moving of input data to train and validation folders finished')

    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in classes:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in classes:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)    

    nb_test_samples = len(os.listdir(os.path.join(test_dir, os.listdir(test_dir)[0])))
    print('total test images:', nb_test_samples )
    
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples
