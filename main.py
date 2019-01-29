import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Prepare the dataset:
mnist = tf.keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = np.asarray(y_train, dtype=np.int32), np.asarray(y_test, dtype=np.int32)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

# Flatten the images
x_train = np.reshape(x_train, [-1, 784])
x_test = np.reshape(x_test, [-1, 784])

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)


# Neural Network
class NeuralNetwork:

    def __init__(self, num_classes, num_outputs):
        # inputs:
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
        self.y = tf.placeholder(dtype=tf.float32)

        # hidden layers:
        h1 = tf.layers.dense(self.x, 50, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 100, activation=tf.nn.relu)
        h3 = tf.layers.dense(h2, 50, activation=tf.nn.relu)

        # dropout layer:
        dropout = tf.layers.dropout(h3, rate=0.02, training=True)

        # output:
        self.y_ = tf.layers.dense(dropout, num_outputs)

        self.choice = tf.argmax(self.y_, axis=1)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.y, tf.int32), depth=num_outputs)
        self.cost = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=self.y_)

        self.accuracy, self.accuracy_op = tf.metrics.accuracy(labels=self.y, predictions=self.choice)

        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.cost, global_step=tf.train.get_global_step())


# TODO: Prepare the dataset batches
BATCH_SIZE = 50
EPOCHS = 5000
load_checkpoint = False

path = "./mnist-nn/"

tf.reset_default_graph()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(buffer_size=y_train.shape[0])
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.repeat()

iterator = dataset.make_initializable_iterator()
features = iterator.get_next()

acc_graph = np.array([])
loss_graph = np.array([])

nn = NeuralNetwork(784, 10)

if not os.path.exists(path):
    os.makedirs(path)

# TODO: Train the model
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

with tf.Session() as sess:

    if load_checkpoint:
        checkpoint = tf.train.get_checkpoint_state(path)
        saver.restore(sess, checkpoint.model_checkpoint_path)
    else:
        sess.run(tf.local_variables_initializer())

    sess.run(init)
    sess.run(iterator.initializer)

    final_accuracy = 0

    for step in range(EPOCHS):
        new_batch = sess.run(features)

        x_batch = new_batch[0]
        y_batch = new_batch[1]

        sess.run((nn.train_op, nn.accuracy_op), feed_dict={nn.x: x_batch, nn.y: y_batch})

        loss, acc = sess.run((nn.cost, nn.accuracy), feed_dict={nn.x: x_batch, nn.y: y_batch})

        final_accuracy = acc

        if step % 10 == 0:
            acc_graph = np.append(acc_graph, acc)
            loss_graph = np.append(loss_graph, loss)

        if step % 100 == 0:
            print("Epoch: {}/{}, Acc: {:0.03f}, Loss: {:0.03f}".format(step, EPOCHS, acc, loss))
            # print("Saving checkpoint")
            # saver.save(sess, path + "mnist", step)

    print("Finished the Training Session with an Accuracy of {:0.03f}".format(final_accuracy))
    print("\nSaving the final checkpoint for the training session.")
    saver.save(sess, path + "mnist", step)

    plt.figure(1, (8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc_graph, 'g-')
    plt.xlabel('steps')
    plt.ylabel('accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss_graph)
    plt.xlabel('steps')
    plt.ylabel('loss')

    test_image = x_test[5000]
    test_label = y_test[5000]

    test_label = np.asarray(test_label, dtype=np.int32)

    plt.figure(2, (4, 4))
    plt.title(test_label)
    plt.imshow(np.reshape(test_image, [28, 28]), cmap="Greys")

    print("Guess: " + str(sess.run(nn.choice, feed_dict={nn.x: np.reshape(test_image, [-1, 784])})))

    plt.show()
