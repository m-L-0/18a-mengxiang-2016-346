# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 10:09:20 2018

@author: 天响之城
"""

import tensorflow as tf


# 导入数据  例子：5.1,3.5,1.4,0.2,Iris-setosa
file = 'iris.txt'
with open(file) as f:
    iris = f.readlines()

# 依照类别分组
setosa = iris[:50]
versicolor = iris[50:100]
virginica = iris[100:]

# 数据平均分
samples = []
for i in range(0, 50):
    samples.append(setosa[i])
    samples.append(versicolor[i])
    samples.append(virginica[i])

# 创建特征及赋值（4个特征）
features = []
labels = []
for sample in samples:
    aux = sample.split(",")
    sepal_length = float(aux[0])
    sepal_width = float(aux[1])
    petal_length = float(aux[2])
    petal_width = float(aux[3])
    label = aux[4].strip()
    features.append([sepal_length, sepal_width, petal_length, petal_width])
    labels.append(label)

# 数据集划分 将鸢尾花数据集按照8 : 2的比例划分成训练集与验证集
percentual_samples_test = 0.8
features_train = features[: int(percentual_samples_test * len(features))]
labels_train = labels[: int(percentual_samples_test * len(features))]
features_test = features[int(percentual_samples_test * len(features)):]
labels_test = labels[int(percentual_samples_test * len(features)):]

#占位符
features_train_placeholder = tf.placeholder("float", [None, 4])
features_test_placeholder = tf.placeholder("float", [4])

#计算L1距离 用L1值寻找最近邻
distance = tf.reduce_sum(tf.abs(tf.add(features_train_placeholder, tf.negative(features_test_placeholder))), reduction_indices=1)
#distance=tf.sqrt(tf.reduce_sumfeatures_train_placeholder, tf.negative(features_test_placeholder)),axis=1))

#分类精确度
accuracy = 0.

# 预测: Get min distance index (K = 1, adjust for other values of K)
pred = tf.arg_min(distance, 0)

# 初始化
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # 运行初始化
    sess.run(init)
    # 遍历测试数据
    for i in range(len(features_test)):
        # 获取当前样本的最近邻索引
        nn_index = sess.run(pred, feed_dict={features_train_placeholder: features_train, features_test_placeholder: features_test[i]})
        # 最近邻分类标签与真实标签比较
        print("Sample", i, " - 预测:", labels_train[nn_index], " /****/ 正确:", labels_test[i])
        # 计算准确率
        if labels_train[nn_index] == labels_test[i]:
            accuracy += 1. / len(features_test)
print("Accuracy:", accuracy)
