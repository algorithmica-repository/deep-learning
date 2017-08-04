# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import tensorflow as tf
import pandas as pd

# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Read the train data
sample = pd.read_csv("train1.csv")
sample.shape
sample.info()

FEATURES = ['x1','x2']
LABEL = ['label']

# input function must return featurecols and labels separately   
def input_function(dataset, train=False):
    feature_cols = {k : tf.constant(dataset[k].values) 
                        for k in FEATURES}
    if train:
        labels = tf.constant(dataset[LABEL].values)
        return feature_cols, labels
    return feature_cols
    
# Build the model with right feature tranformation
feature_cols = [layers.real_valued_column(k) for k in FEATURES]

classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                            n_classes=2,
                                            model_dir="/home/algo/m4")              
classifier.fit(input_fn = lambda: input_function(test,False), steps=1000)

classifier.weights_
classifier.bias_

# Predict the outcome using model
dict = {'x1':[10.4,21.5,10.5], 'x2':[22.1,26.1,2.7] }
test = pd.DataFrame.from_dict(dict)

predictions = classifier.predict(input_fn = lambda: input_function(test,False))
predictions
