#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Spring


import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

train_dir = 'train2/'
logs_train_dir = 'log/'

# 存放用来训练的图片的路径

def get_files(file_dir):
    A5 = []
    label_A5 = []
    A6 = []
    label_A6 = []
    SEG = []
    label_SEG = []
    SUM = []
    label_SUM = []
    LTAX1 = []
    label_LTAX1 = []
    # 定义存放各类别数据和对应标签的列表，列表名对应你所需要分类的列别名
    # A5，A6等是我的数据集中要分类图片的名字


    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'A5':
            A5.append(file_dir + file)
            label_A5.append(0)
        elif name[0] == 'A6':
            A6.append(file_dir + file)
            label_A6.append(1)
        elif name[0] == 'LTAX1':
            LTAX1.append(file_dir + file)
            label_LTAX1.append(2)
        elif name[0] == 'SEG':
            SEG.append(file_dir + file)
            label_SEG.append(3)
        else:
            SUM.append(file_dir + file)
            label_SUM.append(4)
            # 根据图片的名称，对图片进行提取，这里用.来进行划分
            ###这里一定要注意，如果是多分类问题的话，一定要将分类的标签从0开始。这里是五类，标签为0，1，2，3，4。我之前以为这个标签应该是随便设置的，结果就出现了Target[0] out of range的错误。

    print('There are %d A5\nThere are %d A6\nThere are %d LTAX1\nThere are %d SEG\nThere are %d SUM' \
          % (len(A5), len(A6), len(LTAX1), len(SEG), len(SUM)))
    # 打印出提取图片的情况，检测是否正确提取

    image_list = np.hstack((A5, A6, LTAX1, SEG, SUM))
    label_list = np.hstack((label_A5, label_A6, label_LTAX1, label_SEG, label_SUM))
    # 用来水平合并数组

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list
    # 返回两个list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    # tf.cast()用来做类型转换

    input_queue = tf.train.slice_input_producer([image, label])
    # 加入队列

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    # jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # resize

    image = tf.image.per_image_standardization(image)
    # 对resize后的图片进行标准化处理

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch
    # 获取两个batch，两个batch即为传入神经网络的数据


def inference(images, batch_size, n_classes):
    # conv1, shape = [kernel_size, kernel_size, channels, kernel_numbers]
    with tf.variable_scope("conv1") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name="conv1")

    # pool1 && norm1
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope("conv2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding="SAME")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name="conv2")

    # pool2 && norm2
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0,
                          beta=0.75, name='norm2')

    # full-connect1
    with tf.variable_scope("fc1") as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable("weights",
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name="fc1")

    # full_connect2
    with tf.variable_scope("fc2") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name="fc2")

    # softmax
    with tf.variable_scope("softmax_linear") as scope:
        weights = tf.get_variable("weights",
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fc2, weights), biases, name="softmax_linear")
    return softmax_linear


def losses(logits, labels):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=labels, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss


def trainning(loss, learning_rate):
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy


def run_training():

    # 存放一些模型文件的目录
    train, train_label = get_files(train_dir)
    train_batch, train_label_batch = get_batch(train, train_label,
                                               IMG_W,
                                               IMG_H,
                                               BATCH_SIZE,
                                               CAPACITY)
    train_logits = inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = losses(train_logits, train_label_batch)
    train_op = trainning(train_loss, learning_rate)
    train_acc = evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
            if step % 50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%' % (step, tra_loss, tra_acc))
                # 每迭代50次，打印出一次结果
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                # 每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用

    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# 测试一张图片
def get_one_image(img_dir):
    image = Image.open(img_dir)
    # Image.open()
    # 好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
    plt.imshow(image)
    image = image.resize([208, 208])
    image_arr = np.array(image)
    return image_arr


def test(test_file):
    log_dir = 'D:/picture/log/'
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        print(image.shape)
        p = inference(image, 1, 5)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 调用saver.restore()函数，加载训练好的网络模型

                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            print('预测的标签为：')
            print(max_index)
            print('预测的结果为：')
            print(prediction)

            if max_index == 0:
                print('This is a LTAX with possibility %.6f' % prediction[:, 0])
            elif max_index == 1:
                print('This is a SUM with possibility %.6f' % prediction[:, 1])
            elif max_index == 2:
                print('This is a A5 with possibility %.6f' % prediction[:, 2])
            elif max_index == 3:
                print('This is a A6 with possibility %.6f' % prediction[:, 3])
            else:
                print('This is a SEG with possibility %.6f' % prediction[:, 4])


N_CLASSES = 5
# 要分类的类别数，这里是5分类
IMG_W = 208
IMG_H = 208
# 设置图片的size
BATCH_SIZE = 8
CAPACITY = 64
MAX_STEP = 1000
# 迭代一千次，如果机器配置好的话，建议至少10000次以上
learning_rate = 0.0001
# 学习率

BATCH_SIZE = 5
CAPACITY = 64
IMG_W = 208
IMG_H = 208


image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() and i < 2:
            # 提取出两个batch的图片并可视化。
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(BATCH_SIZE):
                print('label: %d' % label[j])
                plt.imshow(img[j, :, :, :])
                plt.show()
            i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    coord.join(threads)
