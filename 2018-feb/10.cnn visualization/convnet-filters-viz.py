from keras.applications import VGG16
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(model, layer_name, filter_index, size=150):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])
    
    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

def viz_filters(model, layers):
    for layer_name in layers:
        size = 64
        margin = 5

        # This a empty (black) image where we will store our results.
        results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

        for i in range(8):  # iterate over the rows of our results grid
            for j in range(8):  # iterate over the columns of our results grid
                # Generate the pattern for filter `i + (j * 8)` in `layer_name`
                filter_img = generate_pattern(model, layer_name, i + (j * 8), size=size)
                
                # Put the result in the square `(i, j)` of the results grid
                horizontal_start = i * size + i * margin
                horizontal_end = horizontal_start + size
                vertical_start = j * size + j * margin
                vertical_end = vertical_start + size
                results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

        # Display the results grid
        plt.figure(figsize=(20, 20))
        plt.imshow(results)
        plt.show()

model = VGG16(weights='imagenet', include_top=False)
layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
layers = ['block1_conv1', 'block2_conv1']
viz_filters(model, layers)
