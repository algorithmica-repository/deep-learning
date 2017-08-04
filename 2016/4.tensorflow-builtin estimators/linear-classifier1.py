# Import required packages
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import os


# Set logging level to info to see detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

os.chdir("/home/algo")
sample = learn.datasets.base.load_csv_with_header(
      filename="train.csv",
      target_dtype=np.int,
      features_dtype=np.float32, target_column=-1)
type(sample)
X_train = sample.data
X_train.shape
type(X_train)

y_train = sample.target
type(y_train)
y_train.shape

#feature_columns argument expects list of tesnorflow feature types
feature_cols = [layers.real_valued_column("", dimension=2)]
                
classifier = learn.LinearClassifier(feature_columns=feature_cols,
                                            n_classes=2,
                                            model_dir="/home/algo/m1")              
classifier.fit(x=X_train, y=y_train, steps=1000)

#access the learned model parameters
classifier.weights_
classifier.bias_

for var in classifier.get_variable_names():
    print classifier.get_variable_value(var)

#w1*x + w2*y + b = 0.
p1 = [0,-classifier.bias_[0]/classifier.weights_[1]]
p2 = [-classifier.bias_[0]/classifier.weights_[0],0]


df = pd.DataFrame(data=np.c_[X_train, y_train.astype(int)], columns=['x1','x2','label'])
sb.swarmplot(x='x1', y='x2', data=df, hue='label', size=10)
plt.plot(p1, p2, 'b-', linewidth = 2)

# Predict the outcome using model
X_test = np.array([[60.4,21.5],[200.1,26.1]])
predictions = classifier.predict(X_test)
predictions
