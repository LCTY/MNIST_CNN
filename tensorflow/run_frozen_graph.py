import os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

GRAPH_NAME = 'quantized_graph.pb'

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            op_dict=None,
            producer_op_list=None
        )
    return graph

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

graph = load_graph('tmp/' + GRAPH_NAME)
for op in graph.get_operations():
    print(op)
    print(op.name)


x = graph.get_tensor_by_name('import/x:0')
y_conv = graph.get_tensor_by_name('import/y_conv:0')

# Add ops to save and restore all the variables.
# since no variables in freeze_graph, not saving is not available
# saver = tf.train.Saver()
with tf.Session(graph=graph) as sess:
    
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")
    # model training
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    train_accuacy = accuracy.eval(feed_dict={x: mnist.test.images,y_: mnist.test.labels})
    print(train_accuacy)
    
        # Save the variables to disk.
    # save_path = "tmp_quantized/"
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # saver.save(sess, save_path+"model.ckpt")
    # print("Model saved in file: %s" % save_path+"model.ckpt")

