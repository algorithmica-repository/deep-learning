import numpy as np
import os
from keras.preprocessing import image
from os.path import join
import json
import re
from PIL import Image

## would work for some structures where you take in a map of the data and sort by filenames
def sort_as_filenames(data, filenames):
    output = []
    for fname in filenames:
        output.append(data[fname])
    return output

def array_to_img(x, dim_ordering='tf', scale=True):
    from PIL import Image
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='tf'):
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

'''
    mappable_extras must be a list of lists. (numpy) Where each sub-list's data will be output as a namedtuple.
'''
class DirectoryIterator(image.Iterator):
    
    def convert_bb(self, item, from_size):
        desired_size = self.target_size
        conv_x = (float(desired_size[0]) / float(from_size[0]))
        conv_y = (float(desired_size[1]) / float(from_size[1]))
        item['height'] = item['height']*conv_y
        item['width'] = item['width']*conv_x
        item['x'] = max(item['x']*conv_x, 0)
        item['y'] = max(item['y']*conv_y, 0)
        return item

    def __init__(self, directory, bbox_directory=None,
                 target_size=(224, 224), color_mode='rgb',
                 dim_ordering='tf',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None):
        self.directory = directory
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering
        if self.color_mode == 'rgb':
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'tf':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

        # first, count the number of samples and classes
        self.nb_sample = 0

        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    classes.append(subdir)
        self.nb_class = len(classes)
        self.class_indices = dict(zip(classes, range(len(classes))))
        #print(classes)
        #print(self.class_indices)

        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.nb_sample += 1
        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        # second, build an index of the images in the different class subfolders
        self.filenames = []
        self.classes = np.zeros((self.nb_sample,), dtype='int32')
        i = 0
        for subdir in classes:
            subpath = os.path.join(directory, subdir)
            for fname in sorted(os.listdir(subpath)):
                is_valid = False
                for extension in white_list_formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    self.classes[i] = self.class_indices[subdir]
                    self.filenames.append(os.path.join(subdir, fname))
                    i += 1
        if bbox_directory:
          self.bbox_directory = bbox_directory
          all_sizes = {f.split('\\')[-1]: Image.open(os.path.join(directory,f)).size for f in self.filenames}
          self.null_largest = {'width':0, 'height': 0, 'x': 224/2., 'y': 224/2.}
          self.file2boxes = {}
          
          boxes = os.listdir(bbox_directory)
          for b in boxes:
              fp = open(os.path.join(bbox_directory, b)) 
              bxs = json.load(fp)
              desired_size = self.target_size
              for item in bxs:            
                  fname = item['filename'].split('/')[-1]
                  if len(item['annotations'])>0:
                      item['annotations'] = [self.convert_bb(a, all_sizes[fname]) for a in item['annotations']]
                      largest = sorted(item['annotations'], key=lambda x: x['height']*x['width'])[-1]
                      #largest =  item['annotations'].sort(key=lambda x: x['width']*x['height'])[-1]
                  else:
                      largest = self.null_largest
                  self.file2boxes[fname] = [largest['width'], largest['height'], largest['x'], largest['y']]

        super(DirectoryIterator, self).__init__(self.nb_sample, batch_size, shuffle, seed)
    

    
    def _get_batches_of_transformed_samples(self, index_array):
        print(index_array)
        labels = []
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype='float32')

        grayscale = self.color_mode == 'grayscale'
        
        if self.bbox_directory:
            boxes = np.zeros((len(batch_x), 4), dtype='float32')
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = load_img(os.path.join(self.directory, fname), grayscale=grayscale, target_size=self.target_size)
            x = img_to_array(img, dim_ordering=self.dim_ordering)
            batch_x[i] = x
    
            if self.bbox_directory:
                null_largest = self.null_largest
                nlarge =  [null_largest['width'], null_largest['height'], null_largest['x'], null_largest['y']]
                f = fname.split('\\')[-1]
                meta = self.file2boxes.get(f)
                if meta == None:
                    boxes[i] = nlarge
                else:
                    boxes[i] = meta
       
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x
        return batch_x, np.hstack((boxes, batch_y))

    def next(self):
        with self.lock:
            index_array = next(self.index_generator)
        return self._get_batches_of_transformed_samples(index_array)