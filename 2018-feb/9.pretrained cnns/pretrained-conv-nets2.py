from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
from keras.models import Model

model1 = VGG16(weights='imagenet', include_top=False)

img_path = 'D:/data/train/cats/cat.1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
features = model1.predict(x)

model2 = Model(inputs=model1.input, outputs=model1.get_layer('block4_pool').output)
block4_pool_features = model2.predict(x)
