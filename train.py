import tensorflow as tf
from alexnet import AlexNet
import cv2
import numpy as np


test_ids = open("./data/test/ids.txt").read().splitlines()
train_ids = open("./data/train/ids.txt").read().splitlines()
whole_ids = {"test": test_ids, "train": train_ids}

train_x = np.zeros((len(train_ids), 512, 512, 3))
test_x = np.zeros((len(test_ids), 512, 512, 3))
train_y = np.zeros((len(train_ids), 32, 32, 1))
test_y = np.zeros((len(test_ids), 32, 32, 1))

cnt = 0
for id in train_ids:
    img_path = "./data/resize/train/" + str(id) + ".jpg"
    mask_path = "./data/mask/train/" + str(id) + ".npy"
    img = cv2.imread(img_path)
    mask = np.load(mask_path)
    train_x[cnt] = img
    train_y[cnt] = mask
    cnt=cnt+1

cnt = 0
for id in test_ids:
    img_path = "./data/resize/test/" + str(id) + ".jpg"
    mask_path = "./data/mask/test/" + str(id) + ".npy"
    img = cv2.imread(img_path)
    mask = np.load(mask_path)
    test_x[cnt] = img
    test_y[cnt] = mask
    cnt=cnt+1


learning_rate = 0.01
num_epochs = 1000
dropout_rate = 0.5

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, 32, 32, 1])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob)
score = model.conv6

'''
print((model.X).get_shape())
print((model.conv1).get_shape())
print((model.conv2).get_shape())
print((model.conv3).get_shape())
print((model.conv4).get_shape())
print((model.conv5).get_shape())
print((model.conv6).get_shape())
'''

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

prediction_int = tf.to_int32(score > 0.6)
prediction = tf.to_float(prediction_int)
correct_prediction = tf.reduce_all(tf.equal(prediction, y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        sess.run(apply_gradients, feed_dict={x: train_x, y: train_y, keep_prob: dropout_rate})
        if epoch%10==0:
			loss, acc = sess.run([cost, accuracy], feed_dict={x: train_x, y: train_y, keep_prob: dropout_rate})
			print("Step: {:5}\tLoss: {:.3f}\tAcc:{:.2%}".format(epoch, loss, acc))
