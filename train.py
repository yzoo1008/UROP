import tensorflow as tf
from alexnet import AlexNet
import cv2
import numpy as np


test_ids = open("./data/test/ids.txt").read().splitlines()
train_ids = open("./data/train/ids.txt").read().splitlines()
whole_ids = {"test": test_ids, "train": train_ids}
train_x = np.array([])
test_x = np.array([])
train_y = np.array([])
test_y = np.array([])

for id in train_ids:
    img_path = "./data/resize/train" + str(id) + ".jpg"
    mask_path = "./data/mask/train" + str(id) + ".jpg"
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    train_x.append(img)
    train_y.append(mask/255)

for id in test_ids:
    mask_path = "./data/mask/test" + str(id) + ".jpg"
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    test_x.append(img)
    test_y.append(mask/255)


learning_rate = 0.01
num_epochs = 1000
dropout_rate = 0.5

x = tf.placeholder(tf.float32, [-1, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, 32, 32, 1])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob)
score = model.conv6

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

prediction = tf.to_int32(score > 0.6)
correct_prediction = tf.reduce_all(tf.equal(prediction, y))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        sess.run(apply_gradients, feed_dict={x: train_x, y: train_y, keep_prob: dropout_rate})
        if epoch%10==0:
            loss, acc = sess.run([cost, accuracy], feed_dict={x: train_x, y: train_y, keep_prob: dropout_rate})
            print("Step: {:5}\tLoss: {:.3f}\tAcc:{:.2%}".format(epoch, loss, acc))
