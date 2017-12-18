import tensorflow as tf


# 数据输入
labels = tf.random_uniform(shape=[3], minval=1, maxval=3, dtype=tf.int32)
predictions = tf.random_uniform(shape=[3], minval=1, maxval=3, dtype=tf.int32)

# 准确率 和 MAE（https://en.wikipedia.org/wiki/Mean_absolute_error）
accuracy, update_op_acc = tf.metrics.accuracy(predictions, labels, name='prediction')
error, update_op_error = tf.metrics.mean_absolute_error(labels, predictions)


with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())

    for batch in range(10):
        accuracy_value, error_value = sess.run([update_op_acc, update_op_error])
        print("iterator: {}, accuracy1: {}, error1: {}".format(batch, accuracy_value, error_value))

    accuracy, mean_absolute_error = sess.run([accuracy, error])
    print("accuracy: {}, mean_absolute_error: {}".format(accuracy, mean_absolute_error))


