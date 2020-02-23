from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Bidirectional
import numpy as np
import keras

k_init = keras.initializers.Constant(value=0.1)
b_init = keras.initializers.Constant(value=0)
r_init = keras.initializers.Constant(value=0.1)

#one length feature vector from lstm
#sequence length=3, no of inputs = 1
input = Input(shape=(3, 1))
features = LSTM(1, return_sequences=True, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(input)
model1 = Model(inputs=input, outputs=features)
data = np.array([0.1, 0.2, 0.3]).reshape((1,3,1))
output = model1.predict(data)
print(output)

#one length feature vector from lstm
#sequence length=3, no of inputs = 1
input = Input(shape=(3, 1))
features = LSTM(1, return_sequences=False, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(input)
model2 = Model(inputs=input, outputs=features)
data = np.array([0.1, 0.2, 0.3]).reshape((1,3,1))
output = model2.predict(data)
print(output)

#one length feature vector from lstm
#sequence length=3, no of inputs = 2
input = Input(shape=(3, 2))
features = LSTM(1, return_sequences=False, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(input)
model3 = Model(inputs=input, outputs=features)
data = np.array([[0.1, 0.2], 
                 [0.3, 0.1],
                 [0.2, 0.3]])
data = data[np.newaxis, ...]
output = model3.predict(data)
print(output)

#two length feature vector from lstm
#sequence length=3, no of inputs = 2
input = Input(shape=(3, 2))
features = LSTM(2, return_sequences=False, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(input)
model4 = Model(inputs=input, outputs=features)
data = np.array([[0.1, 0.2], 
                 [0.3, 0.1],
                 [0.2, 0.3]])
data = data[np.newaxis, ...]
output = model4.predict(data)
print(output)

#two length feature vector from lstm
#sequence length=3, no of inputs = 2
input = Input(shape=(3, 2))
lstm1 = LSTM(2, return_sequences=True, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(input)
features = LSTM(2, return_sequences=False, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init)(lstm1)
model5 = Model(inputs=input, outputs=features)
data = np.array([[0.1, 0.2], 
                 [0.3, 0.1],
                 [0.2, 0.3]])
data = data[np.newaxis, ...]
output = model5.predict(data)
print(output)

#two length feature vector from lstm
#sequence length=3, no of inputs = 2
input = Input(shape=(3, 2))
features = Bidirectional(LSTM(2, return_sequences=False, kernel_initializer=k_init, bias_initializer=b_init, recurrent_initializer=r_init))(input)
model6 = Model(inputs=input, outputs=features)
data = np.array([[0.1, 0.2], 
                 [0.3, 0.1],
                 [0.2, 0.3]])
data = data[np.newaxis, ...]
output = model6.predict(data)
print(output)


