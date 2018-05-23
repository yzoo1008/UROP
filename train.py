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

cnt_train = 0
for id in train_ids:
    img_path = "./data/resize/train/" + str(id) + ".jpg"
    mask_path = "./data/mask_npy/train/" + str(id) + ".npy"
    img = cv2.imread(img_path)
    mask = np.load(mask_path)
    train_x[cnt_train] = img
    train_y[cnt_train] = mask
    cnt_train += 1

cnt_test = 0
for id in test_ids:
    img_path = "./data/resize/test/" + str(id) + ".jpg"
    mask_path = "./data/mask_npy/test/" + str(id) + ".npy"
    img = cv2.imread(img_path)
    mask = np.load(mask_path)
    test_x[cnt_test] = img
    test_y[cnt_test] = mask
    cnt_test += 1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cnt_train
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = cnt_test

num_epochs = 10
batch_size = 64
learning_rate = 0.01

dropout_rate = 0.5

x = tf.placeholder(tf.float32, [None, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, 32, 32, 1])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob)
score = model.conv6

mse = tf.reduce_mean(tf.square(score-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)

'''
NUM_EPOCHES_PER_DECAY = 350.0
lr_decay_factor = 0.1
initial_lr = 0.01

global_step = tf.Variable(0, trainable=False)
num_batches_per_epoch = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)
decay_steps = int(num_batches_per_epoch * NUM_EPOCHES_PER_DECAY)

learning_rate = tf.train.exponential_decay(
		initial_lr,
		global_step,
		decay_steps,
		LEARNING_RATE_DECAY_FACTOR,
		staircase=True)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
gvs = optimizer.compute_gradients(ms)
apply_gradient_op = optimizer.apply_gradients(gvs, global_step=global_step)
'''

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for epoch in range(num_epochs):
		avg_loss = 0.0
		num_truth = 0
		num_predict_truth = 0
		correct_answer = 0.0

		tb = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size)
		for i in range(tb):
			x_batch, y_batch = train_x[i*batch_size:(i+1)*batch_size], train_y[i*batch_size:(i+1)*batch_size]
			feed_dict = {x: x_batch, y: y_batch, keep_prob: dropout_rate}
#			loss, _, sc, lr = sess.run([mse, optimizer, score, learning_rate], feed_dict=feed_dict)
			loss, _, sc = sess.run([mse, optimizer, score], feed_dict=feed_dict)
			avg_loss += loss / tb
			
			for index in range(batch_size):
				for row in range(32):
					for col in range(32):
#						print(sc[index][row][col])

						if sc[index][row][col] > 150.0:
							num_predict_truth += 1
							if y_batch[index][row][col] == 255.0:
								correct_answer += 1
						if y_batch[index][row][col] == 255.0:
							num_truth += 1

#						if sc[index][row][col] != 0.0:
#							print(sc[index][row][col], index, row, col)
				cv2.imwrite('./data/score/train/'+str(epoch)+'/'+str(i*batch_size+index)+'.jpg', sc)
			print("Step: {:5d}\t Num_Batch: {:5d}\tLoss: {:.3f}\t".format(epoch, i, loss))
#						if y_batch[index][i][j] == 255.0:
#							print(y_batch[index][i][j], index, i, j)

		if num_predict_truth == 0:
			print("Step: {:5d}\tLoss: {:.3f}\tRecall: {:.3f}".format(epoch, avg_loss, correct_answer/num_truth))
			print("num_predict_truth: {:5d}".format(num_predict_truth))
		else:
			print("Step: {:5d}\tLoss: {:.3f}\tRecall: {:.3f}\tPrecision: {:.3f}".format(epoch, avg_loss, correct_answer/num_truth, correct_answer/num_predict_truth))
				
