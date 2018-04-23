#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Spring

# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:29:48 2016

@author: root
"""
import tensorflow as tf
import random
import data_utils
import numpy as np

IMAGE_SIZE = 32
CHANNEL = 3
CLASS_NUM = 10

EPOCHS = 1000

BATCH_SIZE = 64
LEARNING_RATE = 0.01
# MAX_STEP = 1000
# TRAIN = False  #
TRAIN = True


def get_data(path='.//cifar-10-batches-py//'):  # cifar10_dir = './/cifar-10-batches-py//'
    return data_utils.get_CIFAR10_data(cifar10_dir=path)


def next_batch(data, iteraction):
    return data[iteraction * BATCH_SIZE:iteraction * BATCH_SIZE + BATCH_SIZE]


def map_to_sparse(data):
    # targets = np.zeros([64, 10], dtype=np.float)
    # for index, value in enumerate(labels):
    #     targets[index, value] = 1.0
    length = len(data)
    res = np.zeros([length, 10], dtype=np.float)
    for i, x in enumerate(data):
        res[i, x] = 1.0
    return res


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)


# 用 get_variable 在 CPU 上定义常量
def variable_on_cpu(name, shape, initializer=tf.constant_initializer(0.1)):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


# 用 get_variable 在 CPU 上定义变量
def variables(name, shape, stddev):
    dtype = tf.float32
    var = variable_on_cpu(name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    return var


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义网络结构
def inference(images):
    with tf.variable_scope('conv1') as scope:
        # 用 5*5 的卷积核，64 个 Feature maps
        weights = variables('weights', [5, 5, 3, 64], 5e-2)
        # 卷积，步长为 1*1
        conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64])
        # 加上偏置
        bias = tf.nn.bias_add(conv, biases)
        # 通过 ReLu 激活函数
        conv1 = tf.nn.relu(bias, name=scope.name)
        # 柱状图总结 conv1
        # tf.summary.histogram(scope.name + '/activations', conv1)
        tf.summary.histogram(scope.name + '/activations', conv1)
    with tf.variable_scope('pooling1_lrn') as scope:
        # 最大池化，3*3 的卷积核，2*2 的卷积
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # 局部响应归一化
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = variables('weights', [5, 5, 64, 64], 5e-2)
        conv = tf.nn.conv2d(norm1, weights, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64])
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        tf.summary.histogram(scope.name + '/activations', conv2)
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')

    with tf.variable_scope('local3') as scope:
        # 第一层全连接
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        dim = reshape.get_shape()[1].value
        weights = variables('weights', shape=[dim, 384], stddev=0.004)
        biases = variable_on_cpu('biases', [384])
        # ReLu 激活函数
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases,
                            name=scope.name)
        # 柱状图总结 local3
        tf.summary.histogram(scope.name + '/activations', local3)

    with tf.variable_scope('local4') as scope:
        # 第二层全连接
        weights = variables('weights', shape=[384, 192], stddev=0.004)
        biases = variable_on_cpu('biases', [192])
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases,
                            name=scope.name)
        tf.summary.histogram(scope.name + '/activations', local4)

    with tf.variable_scope('softmax_linear') as scope:
        # softmax 层，实际上不是严格的 softmax ，真正的 softmax 在损失层
        weights = variables('weights', [192, 10], stddev=1 / 192.0)
        biases = variable_on_cpu('biases', [10])
        softmax_linear = tf.add(tf.matmul(local4, weights), biases,
                                name=scope.name)
        # aa = tf.maximum(softmax_linear.eval, tf.Variable(initial_value=10000, ))
        tf.summary.histogram(scope.name + '/activations', softmax_linear)

    return softmax_linear


# 交叉熵损失层
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        # 交叉熵损失，至于为什么是这个函数，后面会说明。
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/x_entropy', loss)
    return loss


if __name__ == '__main__':
    data = get_data()
    X_train = data["X_train"].transpose(0, 2, 3, 1)

    y_train = data["y_train"]

    X_val = data['X_val'].transpose(0, 2, 3, 1)
    y_val = data['y_val']

    X_test = data['X_test'].transpose(0, 2, 3, 1)
    y_test = data['y_test']

    X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL], name="X")  # 输入的数据占位符
    y_actual = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name="y_actual")  # 输入的标签占位符
    labels = tf.cast(tf.reshape(y_actual, [BATCH_SIZE]), tf.int64)

    logits = inference(X)
    loss = losses(logits, y_actual)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    keep_prob = tf.placeholder("float")

    # global_step 用来设置初始化
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(logits, 1), y_actual)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(EPOCHS):
        iteration = 0
        while iteration < int(X_train.shape[0] / BATCH_SIZE):
            X_batch = next_batch(X_train, iteration)
            y_batch = next_batch(y_train, iteration)  # y_batch = map_to_sparse(y_batch)

            train_op.run(feed_dict={X: X_batch, y_actual: y_batch, keep_prob: 0.8})
            if (i + 1) * iteration % 200 == 0:  # 训练100次，验证一次
                # train_acc = accuracy.eval(feed_dict={x: X_batch, y_actual: y_batch, keep_prob: 1.0})
                train_acc = sess.run([loss, accuracy],
                                     feed_dict={X: X_batch, y_actual: y_batch, keep_prob: 1.0})
                print('epoch :{} step:{}  loss:{} training accuracy:{}'.format(i, iteration, train_acc, ''))
            iteration += 1

    test_accs = 0
    test_its = int(X_test.shape[0] / BATCH_SIZE)
    for i in range(test_its - 1):
        # test_it = random.randint(X_test.shape[0] / BATCH_SIZE - 1)
        test_x_batch = next_batch(X_test, i)
        test_y_batch = next_batch(y_test, i)
        test_acc = accuracy.eval(feed_dict={X: test_x_batch, y_actual: test_y_batch, keep_prob: 1.0})
        test_accs += test_acc

    print("test accuracy ", test_accs / test_its)
