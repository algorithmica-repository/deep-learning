import tensorflow as tf

session = tf.InteractiveSession()

#variable initialized to constant value
a = tf.Variable(10)
print(type(a))
print(a.get_shape())

b = tf.Variable(tf.zeros(5))
print(type(b))
print(b.get_shape())

c = tf.Variable(tf.zeros((2,3)))
print(type(c))
print(c.get_shape())

session.run(tf.initialize_all_variables())
print(a.eval())
print(b.eval())

#variable initialized to random values
tf.set_random_seed(100)
d1 = tf.Variable(tf.random_uniform((10,)))
d2 = tf.Variable(tf.random_uniform((10,),0,2))
d3 = tf.Variable(tf.random_uniform(
                shape=(10,),1,100,dtype=tf.int32))
session.run(tf.initialize_all_variables())
print(d1.eval())
print(d2.eval())
print(d3.eval())

#define  e = d + 20, f = e + 1
d = tf.constant(10)
e = tf.Variable(d+20)
f = tf.add(e, tf.constant(1))
session.run(tf.initialize_all_variables())
print(session.run([d,e,f]))
update = e.assign(e+10)
update.eval()
print(session.run([d,e,f]))

g = tf.Variable(0)
session.run(tf.initialize_all_variables())
for i in range(5):
    update = g.assign(g+1)
    update.eval()
    print(g.eval())

h = tf.Variable(10.0)
type(h)
    
session.close()









