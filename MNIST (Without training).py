from code import *
import imageio


def read_image(path):
    image = imageio.imread(path)
    # image = imageio.imwrite(path, [28, 28])
    image = np.asarray(image)
    image = image / 255.0
    return image


myImage = read_image('./number.jpeg')

print(myImage.shape)


with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(path)
    saver.restore(sess, checkpoint.model_checkpoint_path)

    sess.run(init)

    print(sess.run(nn.choice, feed_dict={nn.x: np.reshape(myImage, [-1, 784])}))
    plt.imshow(myImage, cmap="Greys")
    plt.show()
