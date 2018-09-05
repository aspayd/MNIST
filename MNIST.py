import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)

image_height, image_width = 28, 28

x_train = mnist.train.images
y_train = mnist.train.labels
print(x_train.shape)
print(y_train.shape)

x_test = mnist.train.images
y_test = mnist.train.labels

EPOCHS = 1000
BATCH_SIZE = 50

NUMBER_OF_INPUTS = 784
NUMBER_OF_OUTPUTS = 10
LAYER_1_NODES = 50
LAYER_2_NODES = 100
LAYER_3_NODES = 50

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32, shape=[None, NUMBER_OF_INPUTS])
y_ = tf.placeholder(dtype=tf.float32, shape=[None])

def neural_network(x):
    l1 = tf.layers.dense(x, LAYER_1_NODES, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, LAYER_2_NODES, activation=tf.nn.relu)
    l3 = tf.layers.dense(l2, LAYER_3_NODES, activation=tf.nn.relu)
    dropout = tf.layers.dropout(l3, reate=0.2)
    output = tf.layers.dense(dropout, NUMBER_OF_OUTPUTS, activation=tf.nn.softmax)

    return output


Y = neural_network(X)
choice = tf.nn.softmax(Y)

correct_prediction = tf.equal(tf.argmax(choice, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

onehot = tf.one_hot(indices=tf.cast(y_, dtype=tf.int32), depth=NUMBER_OF_OUTPUTS)
cost = tf.losses.softmax_cross_entropy(onehot, Y)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(EPOCHS):
        batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

        sess.run(train, feed_dict={X: batch_x, y_: batch_y})

        if epoch % 200 == 0:
            acc, loss = sess.run([accuracy, cost], feed_dict={X: batch_x, y_: batch_y})

            print("Epoch: {}/{}, Accuracy: {:.2f}, Loss: {:.2f}".format(epoch, EPOCHS, acc, loss))

    print("Finished Training!\nAcc: {:.2f} Loss: {:.2f}".format(acc, loss))
