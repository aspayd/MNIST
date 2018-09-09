import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data/MNIST", one_hot=True)

image_height, image_width = 28, 28

x_train = mnist.train.images
y_train = np.asarray(mnist.train.labels, dtype=np.int32)
print(x_train.shape)
print(y_train.shape)

x_test = mnist.train.images
y_test = np.asarray(mnist.train.labels, dtype=np.int32)
print(x_test.shape)
print(y_test.shape)

EPOCHS = 1600
BATCH_SIZE = 50

NUMBER_OF_INPUTS = 784
NUMBER_OF_OUTPUTS = 10
LAYER_1_NODES = 256
LAYER_2_NODES = 512
LAYER_3_NODES = 256

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32, shape=[None, NUMBER_OF_INPUTS])
y_ = tf.placeholder(dtype=tf.float32, shape=[None, NUMBER_OF_OUTPUTS])


def neural_network(x):
    l1 = tf.layers.dense(x, LAYER_1_NODES, activation=tf.nn.relu)
    l2 = tf.layers.dense(l1, LAYER_2_NODES, activation=tf.nn.relu)
    l3 = tf.layers.dense(l2, LAYER_3_NODES, activation=tf.nn.relu)
    dropout = tf.layers.dropout(l3, rate=0.2, training=True)
    output = tf.layers.dense(dropout, NUMBER_OF_OUTPUTS)

    return output


Y = neural_network(X)
choice = tf.argmax(Y, axis=1)
probability = tf.nn.softmax(Y)

correct_prediction = tf.equal(tf.argmax(probability, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=Y))

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

            print("Epoch: {}/{}, Accuracy: {:.0f}%, Loss: {:.3f}".format(epoch, EPOCHS, 100*acc, loss))

    print("Finished Training:\nAcc: {:.0f}% Loss: {:.3f}".format(100*acc, loss))