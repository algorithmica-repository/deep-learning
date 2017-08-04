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

sample = learn.datasets.base.load_csv_with_header(
      filename="train.csv",
      target_dtype=np.int,
      features_dtype=np.float32, target_column=-1)

X = sample.data
y = sample.target

# Divide the input data into train and validation
X_train,X_validation,y_train,y_validation = model_selection.train_test_split(X,y, test_size=0.2, random_state=100)
type(X_train)


# creating custom estimator
def model_function(features, targets, mode):
    
  #convert targets to one-hot vector representation   
  targets = tf.one_hot(targets, 2, 1, 0)

  # Configure the single layer perceptron model
  outputs = layers.fully_connected(inputs=features,
                                                 num_outputs=2,
                                                 activation_fn=tf.sigmoid)

  # Calculate loss using mean squared error
  loss = losses.mean_squared_error(outputs, targets)

  # Create an optimizer for minimizing the loss function
  optimizer = layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=0.001,
      optimizer="SGD")

  return {'labels':outputs}, loss, optimizer


#create custom estimator
nn = learn.Estimator(model_fn=model_function, model_dir="/home/algo/m6")

#build the model
nn.fit(x=X_train, y=y_train, steps=2000)
for var in nn.get_variable_names():
    print nn.get_variable_value(var)
    
#evaluate the model using validation set
results = nn.evaluate(x=X_validation, y=y_validation, steps=1)
type(results)
for key in sorted(results):
    print "%s:%s" % (key, results[key])
    
# Predict the outcome of test data using model
X_test = np.array([[100.4,21.5],[200.1,26.1]], dtype=np.float32)
X_test.shape
predictions = nn.predict(X_test)
predictions

