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
BATCH_SIZE = 64
CLASS_NUM = 10

EPOCHS = 50000


def get_data(path='.//cifar-10-batches-py//'):  # cifar10_dir = './/cifar-10-batches-py//'
    return data_utils.get_CIFAR10_data(cifar10_dir=path)


def next_batch(data, iteraction):
    return data[iteraction * BATCH_SIZE:iteraction * BATCH_SIZE + BATCH_SIZE]


def map_to_sparse(data):
    length = len(data)
    res = np.zeros([length, 10])
    for i, x in enumerate(data):
        res[i, x] = 1
    return res


# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape,name=name)
    return tf.Variable(initial_value=initial, dtype=tf.float32, name=name)


# 定义一个函数，用于构建卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 定义一个函数，用于构建池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    data = get_data()
    X_train = data["X_train"].transpose(0, 2, 3, 1)

    y_train = data["y_train"]
    y_train = map_to_sparse(y_train)

    X_val = data['X_val'].transpose(0, 2, 3, 1)
    y_val = data['y_val']
    y_val = map_to_sparse(y_val)

    X_test = data['X_test'].transpose(0, 2, 3, 1)
    y_test = data['y_test']
    y_test = map_to_sparse(y_test)
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  # 下载并加载mnist数据

    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL], name="x")  # 输入的数据占位符
    y_actual = tf.placeholder(tf.float32, shape=[None, CLASS_NUM], name="y_actual")  # 输入的标签占位符


    # 构建网络
    x_image = x  # 转换输入数据shape,以便于用于网络中
    W_conv1 = weight_variable([5, 5, CHANNEL, 32], name="c_w1")
    b_conv1 = bias_variable([32],name="c_b1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
    # h_pool1 = max_pool(h_conv1)  # 第一个池化层

    W_conv2 = weight_variable([5, 5, 32, 64], name="c_w2")
    b_conv2 = bias_variable([64],name="c_b2")
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 第二个卷积层
    # h_pool2 = max_pool(h_conv2)  # 第二个池化层

    W_conv3 = weight_variable([3, 3, 64, 128], name="c_w3")
    b_conv3 = bias_variable([128], name="c_b3")
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)  # 第二个卷积层
    h_pool3 = max_pool(h_conv3)  # 第3个池化层

    W_conv4 = weight_variable([3, 3, 128, 256], name="c_w4")
    b_conv4 = bias_variable([256], name="c_b4")
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)  # 第二个卷积层
    h_pool4 = max_pool(h_conv4)  # 第3个池化层


    # 16 *8 * 8* 64
    w, h, c = h_pool4.shape[1], h_pool4.shape[2], h_pool4.shape[3]
    dimen = int(w * h * c)
    W_fc1 = weight_variable([dimen, 1024], name="fc1_w1")
    b_fc1 = bias_variable([1024], name="fc1_b")
    h_pool2_flat = tf.reshape(h_pool4, [-1, dimen])  # reshape成向量
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

    W_fc2 = weight_variable([1024, 10], name="fc2_w1")
    b_fc2 = bias_variable([10], name="fc2_b")
    y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层
    # tf.nn.sparse_softmax_cross_entropy_with_logits(y, tf.argmax(y_)):
    # y表示的是实际类别，y_表示预测结果
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_actual, 1), logits=y_predict)
    cross_entropy2 = tf.reduce_sum(cross_entropy)  # dont forget tf.reduce_sum()!!

    #cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 交叉熵
    train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy2)  # 梯度下降法

    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    for i in range(EPOCHS):
        iteration = 0
        while iteration < int(X_train.shape[0]/BATCH_SIZE):
            X_batch = next_batch(X_train, iteration)

            y_batch = next_batch(y_train, iteration) #y_batch = map_to_sparse(y_batch)
            train_step.run(feed_dict={x: X_batch, y_actual: y_batch, keep_prob: 0.8})
            if (i + 1) * iteration % 200 == 0:  # 训练100次，验证一次
                #train_acc = accuracy.eval(feed_dict={x: X_batch, y_actual: y_batch, keep_prob: 1.0})
                train_acc = sess.run([cross_entropy2, accuracy], feed_dict={x: X_batch, y_actual: y_batch, keep_prob: 1.0})
                print('epoch :{} step:{}  loss:{} training accuracy:{}'.format(i, iteration, train_acc,''))
            iteration += 1

    test_accs = 0
    test_its = int( X_test.shape[0] / BATCH_SIZE)
    for i in range(test_its - 1):
        # test_it = random.randint(X_test.shape[0] / BATCH_SIZE - 1)
        test_x_batch = next_batch(X_test, i)
        test_y_batch = next_batch(y_test, i)
        test_acc = accuracy.eval(feed_dict={x: test_x_batch, y_actual: test_y_batch, keep_prob: 1.0})
        test_accs += test_acc

    print("test accuracy ", test_accs/test_its)
