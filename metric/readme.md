### Metric

* [tf.metrics](https://www.tensorflow.org/api_docs/python/tf/metrics)  

* [tf.contrib.metrics](https://www.tensorflow.org/api_guides/python/contrib.metrics) 

* [18种和“距离(distance)”、“相似度(similarity)”相关的量的小结](http://blog.csdn.net/solomonlangrui/article/details/47454805)

* [余弦距离、欧氏距离和杰卡德相似性度量的对比分析](https://www.cnblogs.com/chaosimple/archive/2013/06/28/3160839.html)


### streaming metrics
> metrics computed on dynamically valued `Tensors`.
1. Initialization: initializing the metric state.
2. Aggregation: updating the values of the metric state.
3. Finalization: computing the final metric value.


### streaming_mean.py
> [tf.contrib.metrics](https://www.tensorflow.org/api_guides/python/contrib.metrics)
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
> [tf.contrib.metrics](https://www.tensorflow.org/api_guides/python/contrib.metrics)
 * 计算准确率
 * 计算[MAE](https://en.wikipedia.org/wiki/Mean_absolute_error)
 * 用scope解决相同 metric 度量多个不同输入冲突的问题


### tf_accuracy.py
> [tf.metrics](https://www.tensorflow.org/api_docs/python/tf/metrics)
 * 计算准确率


### tf_auc.py
 * AUC
 * ROC
 * PR
 

### tf_confusion_matrix.py
> paper [An introduction to ROC analysis](An%20introduction%20to%20ROC%20analysis.pdf)
 * 分类结果混淆矩阵    
 
 ![分类结果混淆矩阵](confusion_matrix.jpg)


### tf_mean.py
 * 均值相关
 
 
### tf_miou.py
 * mean Intersection-Over-Union
 * mean of the per-class accuracies

