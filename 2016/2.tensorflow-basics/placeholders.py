import tensorflow as tf

session = tf.InteractiveSession()

x = tf.placeholder(tf.int32)
y = tf.placeholder(tf.int32)
z = tf.mul(x,y)
print(session.run(z, feed_dict={x:2,y:3}))

a = tf.placeholder(tf.int32, (2,))
b = tf.placeholder(tf.int32, (2,))
c = tf.add(a, b)
print(session.run(c, feed_dict={a:[10,20], b:[30,40]}))

a = tf.placeholder(tf.int32, (2,2))
b = tf.placeholder(tf.int32, (2,2))
c = tf.add(a, b)
print(session.run(c, feed_dict={a:[[10,20],[30,40]], b:[[1,2],[3,4]] }))

