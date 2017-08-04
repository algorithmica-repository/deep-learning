import tensorflow as tf
import numpy as np

session = tf.InteractiveSession()

#rank0 tensor
ct1 = tf.constant(10)
print(ct1.get_shape())
print(type(ct1))
print(ct1.eval())

#rank 1 tensor
ct2 = tf.constant([10,20,30])
print(ct2.get_shape())
print(ct2.eval())

ct3 = tf.constant(np.array([20,30]))
print(ct3.get_shape())
print(ct3.eval())

#rank 2 tensor
ct4 = tf.constant(np.array([[20,30],[50,60]]))
print(ct4.get_shape())
print(ct4.eval())

ct5 = tf.constant(100, name="constant")
print(ct5.eval())

ct6 = tf.constant(10,shape=[3])
print(ct6.eval())

ct7 = tf.constant(-1,shape=[2,3])
print(ct7.eval())

z1 = tf.zeros(5, tf.int32)
print(type(z1))
print(z1.get_shape())
print(z1.eval())

z2 = tf.zeros((2,2))
print(type(z2))
print(z2.get_shape())
print(z2.eval())

session.close()
