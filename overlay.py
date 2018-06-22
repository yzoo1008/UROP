import cv2
import numpy as np
import os

def npy2img(npy):
	img = np.zeros((32, 32, 3))
	for i in range(32):
		for j in range(32):
			val = npy[i][j]
			if val == 1.:
				img[i][j] = [0, 0, 255]
	return img


ids = open('./test_ids.txt').read().splitlines()
alpha = 0.5

for id in ids:
	image = cv2.imread('./data/resize/test/' + str(id) + '.jpg')
	threshold_npy = np.load('./data/threshold_npy/test/' + str(id) + '.npy')
	threshold = npy2img(threshold_npy)
	threshold_resize = cv2.resize(threshold, (512, 512))
	cv2.imwrite('./test.jpg', threshold_resize)
	overlay = cv2.imread('./test.jpg')
	output = image.copy()

	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	cv2.imshow("Output", output)
	cv2.waitKey(0)
	cv2.imwrite('./result/' + str(id) + '.jpg', output)

os.remove('./test.jpg')
