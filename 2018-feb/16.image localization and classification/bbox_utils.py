from scipy.misc import toimage
import matplotlib.pyplot as plt
import re
from PIL import Image
import os
import json
import numpy as np

def get_file_sizes(directory):
    file2sizes = {}
    for subdir in os.listdir(directory):
       subpath = os.path.join(directory, subdir)
       for fname in os.listdir(subpath):
           tmp = os.path.join(subpath, fname)
           file2sizes[fname] = Image.open(tmp).size
    return file2sizes

def get_bounding_boxes(bbox_directory):                
    null_largest = {'width':0, 'height': 0, 'x': 0, 'y': 0}
    file2boxes = {}
          
    boxes = os.listdir(bbox_directory)
    for b in boxes:
        fp = open(os.path.join(bbox_directory, b)) 
        bxs = json.load(fp)
        for item in bxs:            
            fname = item['filename'].split('/')[-1]
            if len(item['annotations'])>0:
                largest = sorted(item['annotations'], key=lambda x: x['height']*x['width'])[-1]
                largest.pop('class')
            else:
                largest = null_largest
            file2boxes[fname] = largest
    return file2boxes

def convert_bb(box, from_size, desired_size):
        item = box.copy()
        conv_x = (float(desired_size[0]) / float(from_size[0]))
        conv_y = (float(desired_size[1]) / float(from_size[1]))
        item['height'] = item['height']*conv_y
        item['width'] = item['width']*conv_x
        item['x'] = max(item['x']*conv_x, 0)
        item['y'] = max(item['y']*conv_y, 0)
        return item
    
def adjust_bounding_boxes(file2boxes, file2sizes, desired_size):
    f2b = file2boxes.copy()
    for file, box in f2b.items():
        tmp = convert_bb(box, file2sizes[file], desired_size)
        f2b[file] = [tmp['x'], tmp['y'], tmp['height'], tmp['width']]
    return f2b
    
def create_rect(bb, color='red'):
    return plt.Rectangle((bb[0], bb[1]), bb[2], bb[3], color=color, fill=False, lw=3)

def show_bounding_box(data, i):
    img = toimage(data[0][i])
    bbox = data[1][i][0]    
    plt.imshow(img)
    plt.gca().add_patch(create_rect(bbox))

def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]