### Metric

* [tf.metrics](https://www.tensorflow.org/api_docs/python/tf/metrics)  

* [tf.contrib.metrics](https://www.tensorflow.org/api_guides/python/contrib.metrics) 

### streaming metrics
> metrics computed on dynamically valued `Tensors`.

1. Initialization: initializing the metric state.
2. Aggregation: updating the values of the metric state.
3. Finalization: computing the final metric value.


### streaming_mean.py
> [reference](https://www.tensorflow.org/api_guides/python/contrib.metrics)

* Steps：
    1. Initialization: initializing the metric state.
        ```
        mean_value, update_op = tcm.streaming_mean(values, weights=[0.2, 0.8])
        ```
    2. Aggregation: updating the values of the metric state.
        ```
        mean_value_result, value_result = sess.run([update_op, values])
        ```
    3. Finalization: computing the final metric value.
        ```
        final_mean_value_result = mean_value.eval()
        ```


### streaming_accuracy.py
> [reference](https://www.tensorflow.org/api_guides/python/contrib.metrics)
 * 计算准确率
 * 计算[MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)
 * 用scope解决相同 metric 度量多个不同输入冲突的问题


