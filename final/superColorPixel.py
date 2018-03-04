import cv2
import numpy as np
from parameters import *
from image import *
from skimage.segmentation import slic
from graphCut import *
from skimage.segmentation import mark_boundaries

def superColorPixel(img, centers):
	# find superLabels
	superLabels = slic(img.bgr, n_segments=superNumber, compactness=superComp, sigma=superSig)
	# show the output of SLIC
	cv2.namedWindow('superPixel',cv2.WINDOW_NORMAL)
	cv2.imshow('superPixel',mark_boundaries(img.bgr, superLabels))
	cv2.waitKey(0)
	# convert 2D superLabels to a 1D vector
	superLabels = superLabels.flatten()
	# assign new labels
	centDist = pairwiseCost(centers)
	for i in range(superNumber):
		superPixel = img.labels[superLabels==i]
		majorLabel = majorElement(superPixel)
		for x in range(superPixel.shape[0]):
			if centDist[superPixel[x],majorLabel[0]]>superThres:
				superPixel[x] = majorLabel[0]

		img.labels[np.where(superLabels==i)] = superPixel

def majorElement(sp):
	count = np.zeros((K))
	for i in range(K):
		count[i] = np.sum(sp==i)

	return np.flipud(np.argsort(count))