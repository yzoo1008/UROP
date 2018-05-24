import cv2
import numpy as np


test_ids = open("./data/test/ids.txt").read().splitlines()
train_ids = open("./data/train/ids.txt").read().splitlines()
whole_ids = {"test": test_ids, "train": train_ids}

for case in whole_ids:
	for id in whole_ids[case]:
		img_path = "./data/resize/" + case + "/" + str(id) + ".jpg"
		img = cv2.imread(img_path)
		print(img.shape)
		np.save("./data/resize_npy/" + case + "/" + str(id), img)
