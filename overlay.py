import cv2
import numpy as np


def npy2img(npy):
	img = np.zeros((32, 32, 3))
	for i in range(32):
		for j in range(32):
			val = npy[i][j]
			if val == 1.:
				img[i][j] = [0, 0, 255]
	return img


ids = open('./test_ids.txt').read().splitlines()
alpha = 0.4

for id in ids:
	image = cv2.imread('./data/resize/test/' + str(id) + '.jpg')
	threshold_npy = np.load('./data/threshold_npy/test/' + str(id) + '.npy')
	threshold = npy2img(threshold_npy)
	threshold_resize = cv2.resize(threshold, (512, 512))

	overlay = threshold_resize.copy()
	output = threshold_resize.copy()

	cv2.rectangle(overlay, (420, 205), (595, 385), (0, 0, 255), -1)
	cv2.putText(overlay, "alpha={}".format(alpha), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	cv2.imshow("Output", output)
	cv2.waitKey(0)

