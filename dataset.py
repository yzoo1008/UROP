import numpy as np
import cv2


class DataSet:

	def __init__(self, mode='train', in_size = (512, 512), out_size = (32, 32)):

		self.x = []
		self.y = []
		self.pointer = 0
		self.mode = mode
		self.in_size = in_size
		self.out_size = out_size
		self.ids = open('./data/' + mode + '/ids.txt').read().splitlines()
		self.data_size = (len(self.ids))

		self.read_data()


	def read_data(self):

		for id in self.ids:
			img_path = './data/resize/' + str(self.mode) + '/' + str(id) + '.jpg'
			mask_path = './data/mask/' + str(self.mode) + '/' + str(id) + ".jpg"
			self.x.append(img_path)
			self.y.append(mask_path)


	def next_batch(self, batch_size):

		# Get next batch of image (path) and labels
		imgs_path = self.x[self.pointer:self.pointer + batch_size]
		masks_path = self.y[self.pointer:self.pointer + batch_size]

		#update pointer
		self.pointer += batch_size

		# Read images
		images = np.ndarray([batch_size, self.in_size[0], self.in_size[1], 3])
		for i in range(len(imgs_path)):
			img = cv2.imread(imgs_path[i])
			img = img.astype(np.float32)
			images[i] = img

		masks = np.ndarray([batch_size, self.out_size[0], self.out_size[1], 1])
		for i in range(len(masks_path)):
			mask = cv2.imread(masks_path[i])
			mask = mask.astype(np.float32)
			masks[i] = mask

		#return array of images and labels
		return images, masks


	def reset_pointer(self):

		self.pointer = 0

'''
if __name__ == '__main__':
	train_generator = DataSet(mode='train')
	print(train_generator.x[0])
	print(train_generator.ids[0])
	batch_xs, batch_ys = train_generator.next_batch(32)
'''