import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.examples.tutorials.mnist import input_data
from sklearn import metrics
import numpy as np

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
type(mnist)

X_train = mnist.train.images
y_train = mnist.train.labels

X_validation = mnist.validation.images
y_validation = mnist.validation.labels

X_test = mnist.test.images
y_test = mnist.test.labels

# creating custom estimator
def model_function(features, targets, mode):    
    
  #input layer 
  #Reshape features to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel 
  #batch_size corresponds to number of images: -1 represents compute the number of images automatically
  input_layer = tf.reshape(features, [-1, 28, 28, 1]) 
  
  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 28, 28, 1]
  # Output Tensor Shape: [batch_size, 28, 28, 32]
  conv1 = layers.conv2d(
      inputs=input_layer,
      num_outputs=32,
      kernel_size=[5, 5],
      stride=1,
      padding="SAME",
      activation_fn=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 28, 28, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 32]
  pool1 = layers.max_pool2d(inputs=conv1, kernel_size=[2, 2], stride=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 14, 14, 32]
  # Output Tensor Shape: [batch_size, 14, 14, 64]
  conv2 = layers.conv2d(
      inputs=pool1,
      num_outputs=64,
      kernel_size=[5, 5],
      stride=1,
      padding="SAME",
      activation_fn=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = layers.max_pool2d(inputs=conv2, kernel_size=[2, 2], stride=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Fully connected Layers with 100, 20 neurons
  # Input Tensor Shapuntitled0.e: [batch_size, 14 * 14 * 32]
  # Output Tensor Shape: [batch_size, 10]
  fclayers = layers.stack(pool2_flat, layers.fully_connected, [100,20], activation_fn = tf.nn.relu)
  outputs = layers.fully_connected(inputs=fclayers,
                                                 num_outputs=10,
                                                 activation_fn=None)


   # Calculate loss using mean squared error
  loss = losses.softmax_cross_entropy(outputs, targets)

  # Create an optimizer for minimizing the loss function
  optimizer = layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.8,
      optimizer="SGD")

  probs = tf.nn.softmax(outputs)
  return {'probs':probs, 'labels':tf.arg_max(probs,1)}, loss, optimizer


#create custom estimator
nn = learn.Estimator(model_fn=model_function, model_dir="/home/algo/m2")

#build the model
nn.fit(x=X_train, y=y_train, steps=10000, batch_size=100)
for var in nn.get_variable_names():
    print "%s:%s" % (var,nn.get_variable_value(var))
    
# Predict the outcome of test data using model
predictions = nn.predict(X_test, as_iterable=True)
y_pred = []
for i, p in enumerate(predictions):
    y_pred.append(p['labels'])
    print("Prediction %s: %s : %s" % (i + 1, p['probs'], p['labels']))

score = metrics.accuracy_score(np.argmax(y_test,1), y_pred)
score

