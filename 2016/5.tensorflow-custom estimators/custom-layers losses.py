import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from sklearn import model_selection
import numpy as np
import os


# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo")

o = np.array([0,1,2,1,2,1,0])
tmp  = tf.one_hot(o, 3, 1, 0)

x = np.array([10,100,0,-10,-100], dtype=np.float32)
sg = tf.sigmoid(x)

y = np.array([0.6,0.7,0.9,0.9], dtype=np.float32)
sm = tf.nn.softmax(y)

features = np.array([[1,2],[3,4]], dtype=np.float32)

nnout1 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=1,
    activation_fn=None)

nnout2 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=1,
    activation_fn=tf.sigmoid)

nnout3 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([1.0]),
    biases_initializer=tf.constant_initializer([1.0]),
    num_outputs=2,
    activation_fn=None)

nnout4 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=None)

nnout5 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=tf.sigmoid)

nnout6 = layers.fully_connected(
    inputs=features,
    weights_initializer=tf.constant_initializer([[1.0,1.0],[2.0,2.0]]),
    biases_initializer=tf.constant_initializer([1.0,2.0]),
    num_outputs=2,
    activation_fn=tf.nn.softmax)

session = tf.Session()
session.run(tf.initialize_all_variables())
session.run(nnout6)


targets1 = tf.constant([1,1,1,0], dtype=tf.float32)
outputs1 = tf.constant([0,0,0,1], dtype=tf.float32)
sq_loss1 = losses.mean_squared_error(outputs, targets)
log_loss1 = losses.log_loss(outputs, targets)

outputs2 = tf.constant([[100.0, -100.0, -100.0],
                      [-100.0, 100.0, -100.0],
                      [-100.0, -100.0, 100.0]])
targets2 = tf.constant([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]])
sq_loss2 = losses.mean_squared_error(outputs2, targets2)

session.run(sq_loss1)
session.run(log_loss1)

