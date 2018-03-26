import os
import math
import cv2
import random

def get_store_frames(video_file, video_frames_dir):
     print(video_file, video_frames_dir)
     video = cv2.VideoCapture(video_file)
     #print(video.isOpened())
     framerate = video.get(5)
     os.makedirs(video_frames_dir)
     while (video.isOpened()):
         frameId = video.get(1)
         success,image = video.read()
         if(success == False):
             break
         image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
         if (frameId % math.floor(framerate) == 0):
                filename = os.path.join(video_frames_dir, "image_" + str(int(frameId / math.floor(framerate))+1) + ".jpg")
                print(filename)
                cv2.imwrite(filename,image)
     video.release()

def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')
    
    categories = os.listdir(train_dir_original)
    print(categories)

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
       
        for c in categories:
            train_dir_original_cat = os.path.join(train_dir_original, c)
            files = os.listdir(train_dir_original_cat)
            train_files = [os.path.join(train_dir_original_cat, f) for f in files]
            random.shuffle(train_files)    
            n = int(len(train_files) * val_percent)
            val = train_files[:n]
            train = train_files[n:]  
            
            train_category_path = os.path.join(train_dir, c)
            os.mkdir(train_category_path)
            for t in train:
                frames_dir = t.split('\\')[-1].split('.')[0]
                get_store_frames(t, os.path.join(train_category_path, frames_dir))

            val_category_path = os.path.join(validation_dir, c)
            os.mkdir(val_category_path)
            for v in val:
                frames_dir = v.split('\\')[-1].split('.')[0]
                get_store_frames(t, os.path.join(val_category_path, frames_dir))
        
        test_path = os.path.join(test_dir, 'videos')
        os.mkdir(test_path)
        files = os.listdir(test_dir_original)
        test_files = [os.path.join(test_dir_original, f) for f in files]
        for t in test_files:
            frames_dir = t.split('\\')[-1].split('.')[0]
            get_store_frames(t, os.path.join(test_path, frames_dir))
    else:
        print('required directory structure already exists. learning continues with existing data')

    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in categories:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training samples:', nb_train_samples)
    for c in categories:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation samples:', nb_validation_samples)
    nb_test_samples = len(os.listdir(os.path.join(test_dir, 'videos')))
    print('total test samples:', nb_test_samples )
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    preapare_full_dataset_for_flow(
                            train_dir_original='C:\\Users\\data2\\train', 
                            test_dir_original='C:\\Users\\data2\\test',
                            target_base_dir='C:\\Users\\Thimma Reddy\\data2')