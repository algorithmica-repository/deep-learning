import sys
sys.path.append("F:/New Folder/utils")
import math 
import pandas as pd
import numpy as np
import os
from keras.layers import Dense, Input, Conv1D, Flatten
from keras import Model, metrics
import keras_utils as kutils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as m


def rmse(y_true, y_pred):
    y_pred = y_pred.values.reshape(-1)
    y_test = y_true.values.reshape(-1)
    return math.sqrt(m.mean_squared_error(y_test,  y_pred))
   
def format_input_get_feaures_ffnn(df, window_size):
    X_cols = []
    for t in range(window_size):
        col = 'load_t'+str(t+1)
        df[col] = df['load_scaled'].shift(t+1, freq='H')
        X_cols.append(col)
    df.dropna(how='any', inplace=True)
    return X_cols

path = 'F:/'
data = pd.read_excel(os.path.join(path, 'energy.xlsx'))
data.info()
data.head()

#create a time stamp feature with Data and hour together
data['timestamp'] = data['Date'].add(pd.to_timedelta(data.Hour - 1, unit='h'))
data.index.freq = 'H'
data.head()

#filter and rename features
data = data[['timestamp', 'load', 'T']]
data = data.rename(columns={'T':'temp'})
data.head()

#remove the obervations with empty load value
data = data[data.timestamp >= '2006-01-01']
data.head()

#update with timestamp based index
data.index = data['timestamp']
data = data.drop('timestamp', axis=1)
data.head()

#univariate time series data
energy = data[['load']]
energy.head()

#plot the original energy data
energy.plot(y='load', subplots=True, figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

#hold out data for validationa and test data
valid_start_dt = '2014-09-01 00:00:00'
test_start_dt = '2014-11-01 00:00:00'

energy_train = energy[energy.index < valid_start_dt].copy()
energy_validation = energy[(energy.index >= valid_start_dt) & (energy.index < test_start_dt)].copy()
energy_test = energy[energy.index >= test_start_dt].copy()

energy_train[['load']].rename(columns={'load':'train'}) \
    .join(energy_validation[['load']].rename(columns={'load':'validation'}), how='outer') \
    .join(energy_test[['load']].rename(columns={'load':'test'}), how='outer') \
    .plot(y=['train', 'validation', 'test'], figsize=(15, 8), fontsize=12)
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)
plt.show()

#number of lag features
window_size = 10

#preprocess the train data
scaler = MinMaxScaler()
energy_train['load_scaled'] = scaler.fit_transform(energy_train[['load']])
energy_train.head()
#plot load before and after transformation
sns.distplot(energy_train['load'])
sns.distplot(energy_train['load_scaled'])


X_cols = format_input_get_feaures_ffnn(energy_train, window_size)
X_train = energy_train[X_cols].values
X_train = X_train[... , np.newaxis]
y_train = energy_train[['load_scaled']].values

#build and train CNN model
input = Input(shape=(window_size,1))
conv1 = Conv1D(5, kernel_size=2, activation='relu', padding='causal', strides=1, dilation_rate=1)(input)
conv2 = Conv1D(5, kernel_size=2, activation='relu', padding='causal', strides=1, dilation_rate=2)(conv1)
conv3 = Conv1D(5, kernel_size=2, activation='relu', padding='causal', strides=1, dilation_rate=4)(conv2)
features = Flatten()(conv3)
output = Dense(1, activation='linear')(features)

model = Model(inputs=input, outputs=output)
print(model.summary())
model.compile(optimizer='Adam', loss='mse', metrics=[metrics.mean_squared_error])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint(os.path.join(path, "model_{epoch:02d}.h5"), monitor='val_loss', save_best_only=True)   

#preprocess the validation data
energy_validation['load_scaled'] = scaler.transform(energy_validation[['load']])
X_cols = format_input_get_feaures_ffnn(energy_validation, window_size)
X_validation = energy_validation[X_cols].values
X_validation = X_validation[... , np.newaxis]
y_validation = energy_validation[['load_scaled']].values

history = model.fit(x=X_train, y=y_train, verbose=2, epochs=100,  
                    batch_size=32, 
                    validation_data=(X_validation, y_validation),
                    callbacks=[early_stopping, save_model] )
kutils.plot_loss(history)

#preprocess the test data
energy_test['load_scaled'] = scaler.transform(energy_test[['load']])
X_cols = format_input_get_feaures_ffnn(energy_test, window_size)
X_test = energy_test[X_cols].values
X_test = X_test[... , np.newaxis]
y_test = energy_test[['load_scaled']].values

energy_test['load_predicted'] = model.predict(X_test)
energy_test['load_predicted'] = scaler.inverse_transform(energy_test[['load_predicted']])
energy_test[['load', 'load_predicted']].head()
print(rmse(energy_test['load'], energy_test['load_predicted']))

#compare original load vs predicted load
tmp = energy_test[energy_test.index < '2014-11-08']
tmp.plot(x=tmp.index, y=['load', 'load_predicted'], style=['r', 'b'], figsize=(15, 8))
plt.xlabel('timestamp', fontsize=12)
plt.ylabel('load', fontsize=12)