"""
1.mean:mean
2.mae:mean absolute error.
3.mcd:mean cosine distance.
4.mre:mean relative error by normalizing with normalizing with the given values.
5.mse:mean squared error
"""
import tensorflow as tf

labels = tf.random_uniform(shape=[2])
predictions = tf.random_uniform(shape=[2])

# 1.mean:mean
mean, mean_update_op = tf.metrics.mean(values=predictions)

# 2.mae:mean absolute error
# An `absolute_errors` operation computes the absolute value of the differences between `predictions` and `labels`.
mean_absolute_error, mean_absolute_error_update_op = tf.metrics.mean_absolute_error(labels, predictions)

# 3.mcd:mean cosine distance.
# https://www.cnblogs.com/chaosimple/archive/2013/06/28/3160839.html
mean_cosine_distance, mean_cosine_distance_update_op = tf.metrics.mean_cosine_distance(labels, predictions, dim=0)

# 4.mre:mean relative error by normalizing with normalizing with the given values.
# Internally, a `relative_errors` operation divides the absolute value of the differences
#   between `predictions` and `labels` by the `normalizer`.
mean_relative_error, mean_relative_error_update_op = tf.metrics.mean_relative_error(labels, predictions,
                                                                                    normalizer=[.2, .2])
# 5.mse:mean squared error
mean_squared_error, mean_squared_error_update_op = tf.metrics.mean_squared_error(labels, predictions)

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(10):
        mean_relative_error_value = sess.run(mean_relative_error_update_op)
        print("{} mean:{}".format(i, mean_relative_error_value))
        pass
    mean_relative_error_value = sess.run(mean_relative_error)
    print("final mean:{}".format(mean_relative_error_value))
