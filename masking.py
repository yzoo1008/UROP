import numpy as np
import cv2
import math


X = 2048
Y = 2048
X_out = 32
Y_out = 32

test_ids = open("./data/test/ids.txt").read().splitlines()
train_ids = open("./data/train/ids.txt").read().splitlines()
whole_ids = {"test": test_ids, "train": train_ids}

for case in whole_ids:
	for id in whole_ids[case]:
		label_path = "./data/labels/" + case + "/" + str(id) + ".txt"
		label_info = open(label_path).read().splitlines()
		mask = np.zeros((Y_out, X_out, 1))
		for info in label_info:
			_, x_min, y_min, x_max, y_max = info.split(' ')
			if float(x_max)>=2048:
				x_max = 2047.9
			if float(y_max)>=2048:
				y_max = 2047.9
			x_min = int(math.floor(float(x_min)*X_out/X))
			y_min = int(math.floor(float(y_min)*Y_out/Y))
			x_max = int(math.floor(float(x_max)*X_out/X))
			y_max = int(math.floor(float(y_max)*Y_out/Y))

			for y in range(y_min, y_max+1):
				for x in range(x_min, x_max+1):
					mask[y, x] = 255
#cv2.imwrite("./data/mask/" + case + "/" + str(id) + ".jpg", mask)
		np.save("./data/mask_npy/" + case + "/" + str(id), mask)
'''
arr = np.load("./data/mask_npy/train/9998.npy")
print(np.shape(arr))
for i in range(X_out):
	for j in range(Y_out):
		print(arr[i][j])
'''
'''
img_d = cv2.imread("./data/mask/train/9998.jpg")
img = cv2.imread("./data/mask/train/9998.jpg", cv2.IMREAD_GRAYSCALE)
img_reshape = np.reshape(img, (32, 32, 1))
print(np.shape(img_d))
print(np.shape(img))
print(np.shape(img_reshape))
for i in range(X_out):
	for j in range(Y_out):
		print(img[i][j], img_reshape[i][j][0])
print("------------------------------------")
for i in range(X_out):
	for j in range(Y_out):
		print(img_d[i][j])
'''
