#understanding single-layer nn with sigmoid activation + squared error loss
#SGD with eta=0.001

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from sklearn import metrics

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

type(X_train)
X_train.shape
y_train.shape

X_validation.shape
X_test.shape


def explore_data(features, targets):
    randidx = np.random.randint(features.shape[0], size=5)
    for i in randidx:
        curr_img   = np.reshape(features[i, :], (28, 28)) # 28 by 28 matrix 
        curr_label = np.argmax(targets[i, :] ) # Label
        plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
        print ("" + str(i) + "th Training Data " + "Label is " + str(curr_label))
        

explore_data(X_train, y_train)


# creating custom estimator
def model_function(features, targets, mode):    

  # Configure the single layer perceptron model
  outputs = layers.fully_connected(inputs=features,
                                                 num_outputs=10,
                                                 activation_fn=tf.sigmoid)

  # Calculate loss using mean squared error
  loss = losses.mean_squared_error(outputs, targets)

  # Create an optimizer for minimizing the loss function
  optimizer = layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      optimizer="SGD")

  return {'probs':outputs, 'labels':tf.arg_max(outputs,1)}, loss, optimizer


#create custom estimator
nn = learn.Estimator(model_fn=model_function, model_dir="/home/algo/m1")

#build the model
nn.fit(x=X_train, y=y_train, steps=1000, batch_size=100)
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
