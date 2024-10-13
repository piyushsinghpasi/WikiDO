import tensorflow as tf

b = tf.matrix_band_part(tf.ones([256, 256]), -1, 0)

print(b)