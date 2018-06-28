import tensorflow as tf

a = tf.random_uniform([2,3],seed=42)

sess = tf.Session()
sess.run( tf.global_variables_initializer() )

for i in range(1):
    print("a=", sess.run(a))