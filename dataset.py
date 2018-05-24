import numpy as np
import cv2


class DataSet:

	def __init__(self, mode='train'):

		self.x = []
		self.y = []
		self.pointer = 0
		self.mode = mode
		self.ids = open('./data/' + mode + '/ids.txt').read().splitlines()

		if mode == 'train':
			self.ids = "./data/train/ids.txt"
		elif mode == 'test':
			self.ids = "./data/test/ids.txt"
		self.read_data()


	def read_data(self):

		for id in self.ids:
			img_path = './data/resize/' + str(self.mode) + '/' + str(id) + '.jpg'
			mask_path = './data/mask_npy/' + str(self.mode) + '/' + str(id) + ".npy"
			img = cv2.imread(img_path)
			mask = np.load(mask_path)
			self.x.append(img)
			self.y.append(mask)
			self.data_size(len(img))


	def next_batch(self, batch_size):

		# Get next batch of image (path) and labels
		paths = self.images[self.pointer:self.pointer + batch_size]
		labels = self.labels[self.pointer:self.pointer + batch_size]

		#update pointer
		self.pointer += batch_size

		# Read images
		images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
		for i in range(len(paths)):
			img = cv2.imread(paths[i])

			#flip image at random if flag is selected
			if self.horizontal_flip and np.random.random() < 0.5:
				img = cv2.flip(img, 1)

			#rescale image
			img = cv2.resize(img, (self.scale_size[0], self.scale_size[1]))
			img = img.astype(np.float32)

			#subtract mean
			img -= self.mean

			images[i] = img

		# Expand labels to one hot encoding
		one_hot_labels = np.zeros((batch_size, self.n_classes))
		for i in range(len(labels)):
			one_hot_labels[i][labels[i]] = 1

		#return array of images and labels
		return images, one_hot_labels
