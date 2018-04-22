#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @author:Spring

""" this is a cifar 10 image classfiaction model """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import time
import os
import tensorflow as tf

from face_landmark.cnn import data_utils
from argparse import ArgumentParser
from datetime import datetime


batch_size = 32
W = H = 32
C = 3
NUM_CLASSES = 10
KH = 7
KW = 7

# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'  # TOWER_NAME = "TOWER_NAME"

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Whether to float 16 or 32.""")


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint-dir', type=str,
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)


def check_opts(opts):

    def exists(p, msg):
        assert os.path.exists(p), msg

    exists(opts.checkpoint_dir, "checkpoint dir not found!")
    exists(opts.style, "style path not found!")
    exists(opts.train_path, "train path not found!")
    if opts.test or opts.test_dir:
        exists(opts.test, "test img not found!")
        exists(opts.test_dir, "test directory not found!")
    exists(opts.vgg_path, "vgg network data not found!")
    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert os.path.exists(opts.vgg_path)
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0


def get_data(path='.//cifar-10-batches-py//'):  # cifar10_dir = './/cifar-10-batches-py//'
    return data_utils.get_CIFAR10_data(cifar10_dir=path)


def weights_init(shape):
    return tf.Variable(shape, dtype=tf.float32, initial_value=tf.truncated_normal(0.1, 1.0))


def bias_init(shape):
    return tf.Variable(shape, dtype=tf.float32, initial_value=tf.constant(0))


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]

    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _activation_summary(x):
    """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def inference(images):
    """Build the CIFAR-10 model.

  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')

        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(local3)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        _activation_summary(local4)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1 / 192.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


#
def train_fun(total_loss, global_step):
    """Train CIFAR-10 model.

  Create an optimizer and apply to all trainable variables. Add moving
  average for all trainable variables.

  Args:
    total_loss: Total loss from loss().
    global_step: Integer Variable counting the number of training steps
      processed.
  Returns:
    train_op: op for training.
  """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def net():
    X = tf.placeholder(dtype=tf.float32, shape=[batch_size, W, H, C], name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[batch_size, NUM_CLASSES], name="x")

    W1 = weights_init([KH, KW, C, 32])
    b1 = bias_init([32])

    W2 = weights_init([KH, KW, 32, 64])
    b2 = bias_init([64])

    W3 = weights_init([KH, KW, C, 128])
    b3 = bias_init([64])

    # conv layer1  size [batch_size,32,32,3] -> [batch_size, 16, 16, 32]
    net1 = tf.nn.conv2d(X, weights_init([]), padding="VALID") + 1
    tf.nn.batch_normalization(net1)
    net1 = tf.nn.relu(net1)
    net1 = tf.nn.max_pool(net1, ksize=2, strides=1, data_format="NHWC")

    # conv layer2 size [batch_size, 16, 16, 32] ->[batch_size, 8, 8, 64]
    net2 = tf.nn.conv2d(net1, weights_init([]), padding="VALID") + 1
    tf.nn.batch_normalization(net2)
    net2 = tf.nn.relu(net2)
    net2 = tf.nn.max_pool(net2, ksize=2, strides=1, data_format="NHWC")

    # conv layer3 size [batch_size, 8, 8, 64] -> [batch_size, 4, 4, 128]
    net3 = tf.nn.conv2d(net2, weights_init([]), padding="VALID") + 1
    tf.nn.batch_normalization(net3)
    net3 = tf.nn.relu(net3)
    net3 = tf.nn.max_pool(net3, ksize=2, strides=1, data_format="NHWC")

    # FC  [batch_size, 4, 4, 128]
    net3.shape
    W_FC1 = weights_init([KH, KW, 32, 64])
    b_FC1 = bias_init([64])


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def loss_fun(logits, labels):
    """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:  #with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()
        # cifar10.maybe_download_and_extract()
        print(" process data start ")
        data = get_data()
        X_train = data["X_train"]
        y_train = data["y_train"]

        X_val = data['X_val']
        y_val = data['y_val']

        X_test = data['X_test']
        y_test = data['y_test']

        # with tf.Graph().as_default():
        #     global_step = tf.train.get_or_create_global_step()
        #     with tf.device('/cpu:0'):
        #         images, labels = X_train, y_train
        #         inference(images)

        # Get images and labels for CIFAR-10.
        # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
        # GPU and resulting in a slow down.
        epochs = 5
        # images, labels = cifar10.distorted_inputs()
        images, labels = X_train, y_train
        X_ = tf.placeholder(tf.float32, shape=[batch_size,32,32,3], name='X_')
        y_ = tf.placeholder(tf.float32, shape=[batch_size], name='y_')
        #x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符

        for epoch in range(epochs):
            num_examples = X_train.shape[0]

            iterations = 0
            while iterations * batch_size < num_examples:
                start_time = time.time()
                curr = iterations * batch_size
                step = curr + batch_size
                X_batch = images[iterations*batch_size:iterations*batch_size+batch_size]
                y_batch = labels[iterations*batch_size:iterations*batch_size+batch_size]

                iterations += 1
                assert X_batch.shape[0] == batch_size

                # Build a Graph that computes the logits predictions from the
                # inference model.
                logits = inference(X_)

                # Calculate loss.
                loss = loss_fun(logits, labels)

                # Build a Graph that trains the model with one batch of examples and
                # updates the model parameters.
                train_op = train_fun(loss, global_step)

                class _LoggerHook(tf.train.SessionRunHook):
                    """Logs loss and runtime."""

                    def begin(self):
                        self._step = -1
                        self._start_time = time.time()

                    def before_run(self, run_context):
                        self._step += 1
                        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

                    def after_run(self, run_context, run_values):
                        if self._step % FLAGS.log_frequency == 0:
                            current_time = time.time()
                            duration = current_time - self._start_time
                            self._start_time = current_time

                            loss_value = run_values.results
                            examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                            sec_per_batch = float(duration / FLAGS.log_frequency)

                            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                          'sec/batch)')
                            print(format_str % (datetime.now(), self._step, loss_value,
                                                examples_per_sec, sec_per_batch))

                with tf.train.MonitoredTrainingSession(
                        checkpoint_dir=FLAGS.train_dir,
                        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                               tf.train.NanTensorHook(loss),
                               _LoggerHook()],
                        config=tf.ConfigProto(
                            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
                    while not mon_sess.should_stop():
                        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
