from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

#get the whole vgg16 model
model = VGG16()
print(type(vgg_model))
print(vgg_model.summary())
for layer in vgg_model.layers:
    print(layer.name, layer.input, layer.output)
print(vgg_model.input)
print(vgg_model.output)

img_path = 'D:/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print(np.argmax(preds, axis=1))
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])

#get the vgg16 model without top fc layers
model = VGG16(include_top=False)
print(type(vgg_model))
print(vgg_model.summary())
for layer in vgg_model.layers:
    print(layer.name, layer.input, layer.output)
print(vgg_model.input)
print(vgg_model.output)

img_path = 'D:/elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)