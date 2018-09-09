import tensorflow as tf
from tensorflow import keras
import numpy as np

mnist = keras.datasets.mnist.load_data()

(x_train, y_train), (x_test, y_test) = mnist

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = np.reshape(x_train, (-1, 784))
x_test = np.reshape(x_test, (-1, 784))

EPOCHS = 5
BATCH_SIZE = 64
INPUTS = 784
OUTPUTS = 10
L1_NODES = 50
L2_NODES = 100
L3_NODES = 50


def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(INPUTS, activation=tf.nn.relu))
    model.add(keras.layers.Dense(L1_NODES, activation=tf.nn.relu))
    model.add(keras.layers.Dense(L2_NODES, activation=tf.nn.relu))
    model.add(keras.layers.Dense(L3_NODES, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.Dense(OUTPUTS, activation=tf.nn.softmax))

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=['accuracy']
    )

    return model


model = create_model()
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
