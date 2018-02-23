import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
import pandas as pd
from scipy.misc import imread

def one_hot(a, N):
    b = np.zeros((a.size, N))
    b[np.arange(a.size), a] = 1
    return b

# TODO: Load traffic signs data.
data = np.load('train.p')
print(data.keys())
# for k, w in data.items():
#     print(k, w.shape, w[0])

# read csv
signames = pd.read_csv('signnames.csv')
class_names = list(signames.SignName)
n_class = len(class_names)
# print(signames, len(signames))

# TODO: Split data into training and validation sets.
X_train, X_validate, y_train, y_validate = train_test_split(data['features'], data['labels'])
print(X_train.shape, X_validate.shape, y_train.shape, y_validate.shape)
y_train = one_hot(y_train, n_class)
y_validate = one_hot(y_validate, n_class)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
resized = tf.image.resize_images(x, (227, 227))
Y = tf.placeholder(tf.int64, [None, n_class])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
# fc7 size is 4096
# n_class is 43
mu = 0
sigma = 0.1
input_N = fc7.get_shape().as_list()[-1] 
fc8W = tf.Variable(tf.truncated_normal((input_N, n_class), mean = mu, stddev = sigma))
fc8b = tf.Variable(tf.zeros(n_class))

logits = tf.matmul(fc7, fc8W) + fc8b
prediction = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
# Define loss and optimizer
rate = 0.001
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

# TODO: Train and evaluate the feature extraction model.
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

BATCH_SIZE = 128
def evaluate(sess, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, Y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Run Inference
import time
t = time.time()
# Read Images
im1 = imread("construction.jpg").astype(np.float32)
im1 = im1 - np.mean(im1)

im2 = imread("stop.jpg").astype(np.float32)
im2 = im2 - np.mean(im2)

input = [im1, im2]

accuracy = 0
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
output = sess.run(prediction, feed_dict={x: input})
# accuracy = evaluate(sess, X_validate, y_validate)
sess.close()

# Print Output
for input_im_ind in range(output.shape[0]):
    inds = np.argsort(output)[input_im_ind, :]
    print("Image", input_im_ind)
    for i in range(5):
        print("%s: %.3f" % (class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]]))
    print()


print("Time: %.3f seconds, accuracy: %.3f" % (time.time() - t, accuracy))
