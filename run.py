import os
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from dataset import DataSet


# Learning params
learning_rate = 0.01
num_epochs = 10
batch_size = 32

# Network params
dropout_rate = 0.5
num_classes = 2
train_layers = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "./tf_board/mask"
checkpoint_path = "./tf_board/"

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# TF placeholder for graph input and output
x = tf.placeholder(tf.float32, [batch_size, 512, 512, 3])
y = tf.placeholder(tf.float32, [None, 32, 32, 1])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, train_layers)

# Link variable to model output
score = model.conv8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
	loss = tf.reduce_mean(tf.square(score - y))
#	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))

# Train op
with tf.name_scope("train"):
	# Get gradients of all trainable variables
	gradients = tf.gradients(loss, var_list)
	gradients = list(zip(gradients, var_list))

	# Create optimizer and apply gradient descent to the trainable variables
	optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary  
for gradient, var in gradients:
	tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary  
for var in var_list:
	tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("recall"):
	x_threshold = tf.to_int32(score >= 100.)
	y_threshold = tf.to_int32(y >= 255.)

	num_truth = tf.to_float32(tf.reduce_sum(y_threshold))
	num_correct = tf.to_float32(tf.reduce_sum(tf.multiply(x_threshold, y_threshold)))

	if num_truth == 0.:
		recall = tf.constant(0., dtype = tf.float32)
	else:
		recall = tf.div(num_correct, num_truth)

with tf.name_scope("precision"):
	x_threshold = tf.to_int32(score >= 100.)
	y_threshold = tf.to_int32(y >= 255.)

	num_correct = tf.to_float32(tf.reduce_sum(tf.multiply(x_threshold, y_threshold)))
	num_predict = tf.to_float32(tf.reduce_sum(x_threshold))

	if num_predict == 0.:
		precision = tf.constant(0., dtype= tf.float32)
	else:
		precision = tf.div(num_correct, num_predict)


tf.summary.scalar('recall', recall)
tf.summary.scalar('precision', precision)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Initalize the data generator seperately for the train and test set
train_generator = DataSet(mode='train')
test_generator = DataSet(mode='test')

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

	# Loop over number of epochs
	for epoch in range(num_epochs):

		print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

		step = 1

		while step < train_batches_per_epoch:

			# Get a batch of images and labels
			batch_xs, batch_ys = train_generator.next_batch(batch_size)

			# And run the training op
			sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout_rate})

			# Generate summary with the current batch of data and write to file
			if step % display_step == 0:
				s = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
				writer.add_summary(s, epoch * train_batches_per_epoch + step)

			step += 1

		# Test the model on the entire test set
		print("{} Start test".format(datetime.now()))
		test_rec = 0.
		test_pre = 0.
		test_count = 0
		for _ in range(test_batches_per_epoch):
			batch_tx, batch_ty = test_generator.next_batch(batch_size)
			rec, pre = sess.run([recall, precision], feed_dict={x: batch_tx, y: batch_ty, keep_prob: 1.})
			test_rec += rec
			test_pre += pre
			test_count += 1
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
