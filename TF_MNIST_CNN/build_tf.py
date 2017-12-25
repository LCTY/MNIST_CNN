import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def read_weight(shape, var_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=var_name)


def weight_varible(shape, var_name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=var_name)


def bias_variable(shape, var_name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=var_name)


def conv2d(x, W, conv2d_name=None):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=conv2d_name)


def max_pool(x, max_pool_name=None):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=max_pool_name)


# First model.
def build_model_1():
    # paras
    w_conv1 = weight_varible([3, 3, 1, 32], "w_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    # conv layer-1
    _x = tf.placeholder(tf.float32, [None, 784], name="x")
    x_image = tf.reshape(_x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 'conv2d1') + b_conv1, name="h_conv1")
    h_pool1 = max_pool(h_conv1, 'max_pool1')

    # conv layer-2
    w_conv2 = weight_varible([3, 3, 32, 64], "w_conv2")
    b_conv2 = bias_variable([64], "b_conv2")

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 'conv2d2') + b_conv2, name="h_conv2")
    h_pool2 = max_pool(h_conv2, 'max_pool2')

    # full connection
    w_fc1 = weight_varible([7 * 7 * 64, 1024], "w_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="h_fc1")

    # dropout
    _keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, _keep_prob, name="h_fc1_drop")

    # output layer: softmax
    w_fc2 = weight_varible([1024, 10], "w_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y_conv")
    _y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # model training
    cross_entropy = -tf.reduce_sum(_y_ * tf.log(y_conv))
    _train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(_y_, 1))
    _accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return _x, _y_, _train_step, _accuracy, _keep_prob, "model_1"


# Hipsternet model.
def build_model_2():
    # paras
    w_conv1 = weight_varible([3, 3, 1, 10], "w_conv1")
    b_conv1 = bias_variable([10], "b_conv1")

    # conv layer-1
    _x = tf.placeholder(tf.float32, [None, 784], name="x")
    x_image = tf.reshape(_x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 'conv2d1') + b_conv1, name="h_conv1")
    h_pool1 = max_pool(h_conv1, 'max_pool1')

    # full connection
    w_fc1 = weight_varible([14 * 14 * 10, 128], "w_fc1")
    b_fc1 = bias_variable([128], "b_fc1")

    h_pool1_flat = tf.reshape(h_pool1, [-1, 14 * 14 * 10])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1, name="h_fc1")

    # dropout
    _keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, _keep_prob, name="h_fc1_drop")

    # output layer: softmax
    w_fc2 = weight_varible([128, 10], "w_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y_conv")
    _y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # model training
    cross_entropy = -tf.reduce_sum(_y_ * tf.log(y_conv))
    _train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(_y_, 1))
    _accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return _x, _y_, _train_step, _accuracy, _keep_prob, "model_2"


# Smaller model_1.
def build_model_3():
    # paras
    w_conv1 = weight_varible([3, 3, 1, 10], "w_conv1")
    b_conv1 = bias_variable([10], "b_conv1")

    # conv layer-1
    _x = tf.placeholder(tf.float32, [None, 784], name="x")
    x_image = tf.reshape(_x, [-1, 28, 28, 1])

    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 'conv2d1') + b_conv1, name="h_conv1")
    h_pool1 = max_pool(h_conv1, 'max_pool1')

    # conv layer-2
    w_conv2 = weight_varible([3, 3, 10, 20], "w_conv2")
    b_conv2 = bias_variable([20], "b_conv2")

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 'conv2d2') + b_conv2, name="h_conv2")
    h_pool2 = max_pool(h_conv2, 'max_pool2')

    # full connection
    w_fc1 = weight_varible([7 * 7 * 20, 128], "w_fc1")
    b_fc1 = bias_variable([128], "b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 20])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name="h_fc1")

    # dropout
    _keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, _keep_prob, name="h_fc1_drop")

    # output layer: softmax
    w_fc2 = weight_varible([128, 10], "w_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name="y_conv")
    _y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # model training
    cross_entropy = -tf.reduce_sum(_y_ * tf.log(y_conv))
    _train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(_y_, 1))
    _accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return _x, _y_, _train_step, _accuracy, _keep_prob, "model_3"


def build_model_4():
    # paras
    w_conv1 = weight_varible([3, 3, 1, 32], "w_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    # conv layer-1
    _x = tf.placeholder(tf.float32, [None, 784], name="x")
    x_image = tf.reshape(_x, [-1, 28, 28, 1])

    h_conv1_tmp = tf.add(conv2d(x_image, w_conv1, 'conv2d1'), b_conv1, name="h_conv1_tmp")
    h_conv1 = tf.nn.relu(h_conv1_tmp, name="h_conv1")
    h_pool1 = max_pool(h_conv1, 'max_pool1')

    # conv layer-2
    w_conv2 = weight_varible([3, 3, 32, 32], "w_conv2")
    b_conv2 = bias_variable([32], "b_conv2")

    h_conv2_tmp = tf.add(conv2d(h_pool1, w_conv2, 'conv2d2'), b_conv2, name="h_conv2_tmp")
    h_conv2 = tf.nn.relu(h_conv2_tmp, name="h_conv2")
    h_pool2 = max_pool(h_conv2, 'max_pool2')
    # h_pool3 = max_pool(h_pool2, 'max_pool3')

    # full connection
    w_fc1 = weight_varible([7 * 7 * 32, 512], "w_fc1")
    b_fc1 = bias_variable([512], "b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 32])
    h_fc1_tmp = tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1, name="h_fc1_tmp")
    h_fc1 = tf.nn.relu(h_fc1_tmp, name="h_fc1")

    # dropout
    _keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    h_fc1_drop = tf.nn.dropout(h_fc1, _keep_prob, name="h_fc1_drop")

    # output layer: softmax
    w_fc2 = weight_varible([512, 10], "w_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv_tmp = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2, name="y_conv_tmp")
    y_conv = tf.nn.softmax(y_conv_tmp, name="y_conv")
    _y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # model training
    cross_entropy = -tf.reduce_sum(_y_ * tf.log(y_conv))
    _train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(_y_, 1))
    _accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return _x, _y_, _train_step, _accuracy, _keep_prob, "model_4"


# Read MNIST database.
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("Download Done!")

# Build the model.
x, y_, train_step, accuracy, keep_prob, model_name = build_model_4()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, and save the variables to disk.
with tf.Session() as sess:
    # Initialize the variables.
    sess.run(tf.global_variables_initializer())

    batch_size = 64
    train_iter = 20000
    one_tenth = int(train_iter * 0.1)

    # Train the model.
    end = 0
    for i in range(train_iter):
        batch = mnist.train.next_batch(batch_size)
        if i % one_tenth == 0:
            idx_start, idx_end = (i % one_tenth) * batch_size, ((i % one_tenth) + 1) * batch_size
            train_accuracy = accuracy.eval(feed_dict={
                x: mnist.test.images[idx_start:idx_end],
                y_: mnist.test.labels[idx_start:idx_end],
                keep_prob: 1.0
            })
            print("step %d, training accuracy %g" % (i, train_accuracy))
            """ 
            # stop if accuracy high enough
            if train_accuracy >= 0.999:
                end += 1
                if end >= 5:
                    break
            else:
                end = 0
            """
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    # Save the variables to disk.
    save_path = "data/" + model_name + "/weight/tf"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    saver.save(sess, (save_path + "/model.ckpt"))
    tf.train.write_graph(sess.graph_def, save_path, "model_age.pb", as_text=True)
    print("Model saved in file: %s" % (save_path + "/model.ckpt"))
