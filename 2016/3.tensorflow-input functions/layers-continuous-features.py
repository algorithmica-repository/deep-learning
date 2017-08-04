from tensorflow.contrib import layers
import pandas as pd
import tensorflow as tf

dict = {'id':range(1,6), 'pclass':[1,2,1,2,1], 'gender':['M','F','F','M','M'], 
        'fare':[10.5,22.3,11.6,22.4,31.5] }
df = pd.DataFrame.from_dict(dict)
df.shape
df.info()

#real valued column
id = layers.real_valued_column('id')
type(id)
id.key

#real valued column
fare = layers.real_valued_column('fare')
type(fare)
fare.key

cont_features = ['id','fare']

#comprehension for creating all real valued columns once
cont_feature_cols = [layers.real_valued_column(k) for k in cont_features]

#bucketized column
fare_buckets = layers.bucketized_column(fare, boundaries=[15,30])
type(fare_buckets)
fare_buckets.key


#converting continuous valued feature data to constant tensor
type(df[['id']])
df[['id']].size
type(df[['id']].values)
ct = tf.constant(df[['id']].values)
type(ct)

cont_features_tensor = {k : tf.constant(df[k].values) 
                            for k in cont_features }

