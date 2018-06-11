import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from dataset import DataSet


# Learning params
initial_learning_rate = 0.0000005
num_epochs = 20
batch_size = 32

# Network params
dropout_rate = 0.5
train_layers = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tf_board/mask"
checkpoint_path = "./tf_board/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# Initalize the data generator seperately for the train and test set
train_generator = DataSet(mode='train')
test_generator = DataSet(mode='test')
train_size = train_generator.data_size

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, 32, 32, 1])
keep_prob = tf.placeholder(tf.float32)
batch_step = tf.placeholder(tf.int32)

# Initialize model
model = AlexNet(x, keep_prob, train_layers)

# Link variable to model output
score = model.conv8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
ground_truth_true = tf.to_float(tf.to_int32(y >= 1.))
ground_truth_false = tf.to_float(tf.to_int32(y <= -1.))

num_ground_truth_true = tf.reduce_sum(ground_truth_true)
num_ground_truth_false = tf.reduce_sum(ground_truth_false)

# 1)
weight_t = tf.div(num_ground_truth_false, num_ground_truth_true)
weight_t_map = tf.multiply(weight_t, ground_truth_true)
se = tf.square(score - y)
compensate_true = tf.multiply(weight_t_map, se)
f_map = tf.multiply(ground_truth_false, se)

# 2)
ground_truth_true_reshape = tf.reshape(ground_truth_true, [-1])
shuffle= tf.random_shuffle(ground_truth_true_reshape)
shuffle_map = tf.reshape(shuffle, tf.shape(ground_truth_true))
random_pick = tf.multiply(ground_truth_false, shuffle_map)              # pick num of true grids in false grids.

score_false = tf.multiply(score, random_pick)
score_true = tf.multiply(score, ground_truth_true)
score_total = tf.add(score_false, score_true)                           # non-interesting regions are 0.
y_false = tf.multiply(ground_truth_false, random_pick)
y_total = tf.add(y_false, ground_truth_true)

with tf.name_scope("cross_ent"):
	# default)
	# loss = tf.reduce_mean(tf.reduce_sum(tf.square(score-y)))
	#  1)
	loss = tf.reduce_sum(tf.add(compensate_true, f_map))
	#  2)
	# loss = tf.reduce_sum(tf.square(score_total - y_total))

# Train op
with tf.name_scope("train"):
	# Get gradients of all trainable variables
	gradients = tf.gradients(loss, var_list)
	gradients = list(zip(gradients, var_list))

	# Create optimizer and apply gradient descent to the trainable variables
	learning_rate = tf.train.exponential_decay(initial_learning_rate, batch_step*batch_size, train_size, 0.8, staircase=True)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	# train_op = optimizer.apply_gradients(grads_and_vars=gradients)

	# 2)
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Add gradients to summary  
for gradient, var in gradients:
	tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
	tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)
tf.summary.scalar('learning_rate', learning_rate)

x_threshold = tf.to_int32(score >= 0.6)
y_threshold = tf.to_int32(y >= 1.)
num_truth = tf.to_float(tf.reduce_sum(y_threshold))
num_correct = tf.to_float(tf.reduce_sum(tf.multiply(x_threshold, y_threshold)))
num_predict = tf.to_float(tf.reduce_sum(x_threshold))

with tf.name_scope("recall"):
	recall = tf.cond(num_truth > 0., lambda: tf.div(num_correct, num_truth), lambda: tf.constant(0., dtype = tf.float32))

with tf.name_scope("precision"):
	precision = tf.cond(num_predict > 0., lambda: tf.div(num_correct, num_predict), lambda: tf.constant(0., dtype = tf.float32))


tf.summary.scalar('recall', recall)
tf.summary.scalar('precision', precision)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the number of train/test steps per epoch
train_batches_per_epoch = np.floor(train_generator.data_size / batch_size).astype(np.int16)
test_batches_per_epoch = np.floor(test_generator.data_size / batch_size).astype(np.int16)

# Start Tensorflow session
with tf.Session() as sess:
	# Initialize all variables
	sess.run(tf.global_variables_initializer())

	# Add the model graph to TensorBoard
	writer.add_graph(sess.graph)

	# Load the pretrained weights into the non-trainable layer
	model.load_initial_weights(sess)

	print("{} Start training...".format(datetime.now()))
	print("{} Open Tensorboard at --logdir {}".format(datetime.now(), filewriter_path))

	total_step = 1
	# Loop over number of epochs
	for epoch in range(num_epochs):

		print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

		step = 1

		while step < train_batches_per_epoch:

			# Get a batch of images and labels
			batch_xs, batch_ys = train_generator.next_batch(batch_size)

			# And run the training op
			n_t, n_c, n_p, cost, lr, _ = sess.run([num_truth, num_correct, num_predict, loss, learning_rate, train_op],
			                                           feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate, batch_step: total_step})
			print("Epoch: {:.0f}/{:.0f}\tStep: {:.0f}/{:.0f}\tTrue: {:.0f}\tCorr: {:.0f}\tPred: {:.0f}\tLoss: {:.5f}\tLr: {:.9f}"
			      .format(epoch+1, num_epochs, step, train_batches_per_epoch, n_t, n_c, n_p, cost, lr))

			# Generate summary with the current batch of data and write to file
			if step % display_step == 0:
				s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1., batch_step: total_step})
				writer.add_summary(s, epoch * train_batches_per_epoch + step)

			step += 1
			total_step += 1

		# Test the model on the entire test set
		print("{} Start test".format(datetime.now()))
		test_rec = 0.
		test_pre = 0.
		test_count = 0
		for _ in range(test_batches_per_epoch):
			batch_tx, batch_ty = test_generator.next_batch(batch_size)
			rec, pre, truth, correct, predict = sess.run([recall, precision, num_truth, num_correct, num_predict], feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1., batch_step: total_step})
			test_rec += rec
			test_pre += pre
			test_count += 1
			print("True: {:.0f}\t Corr: {:.0f}\t Pred: {:.0f}".format(truth, correct, predict))
		test_rec /= test_count
		test_pre /= test_count
		print("{} Test Recall = {:.4f}\t Precision = {:.4f}".format(datetime.now(), test_rec, test_pre))

		# Reset the file pointer of the image data generator
		test_generator.reset_pointer()
		train_generator.reset_pointer()

		print("{} Saving checkpoint of model...".format(datetime.now()))

		# save checkpoint of the model
		checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
		save_path = saver.save(sess, checkpoint_name)

		print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
