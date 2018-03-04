import cv2
import pygco
import pyqpbo
import numpy as np
from parameters import *
from image import *

def linkWeight(img):
	link = []
	contourImg = edgeDetect(img.L)
	contourImg = (contourImg+10) / np.max(contourImg)
	for i in range(0, img.H):
		for j in range(0, img.W):
			nowIndex = i*img.W + j
			if i-1>=0 : # now & up(nowIndex-W)
				weight = computeWeight(i, j, i-1, j, contourImg)
				link.append([nowIndex, nowIndex-img.W, weight])				
			if i+1<img.H : # now & down(nowIndex+W)
				weight = computeWeight(i, j, i+1, j, contourImg)
				link.append([nowIndex, nowIndex+img.W, weight])
			if j-1>=0 : # now & left(nowIndex-1)
				weight = computeWeight(i, j, i, j-1, contourImg)
				link.append([nowIndex, nowIndex-1, weight])
			if j+1<img.W : # now & right(nowIndex+1)
				weight = computeWeight(i, j, i, j+1, contourImg)
				link.append([nowIndex, nowIndex+1, weight])

	link = np.array(link).astype('int32')
	return link

def edgeAndWeight(img, pairwise):
	edgeWeight = []
	edge = []
	contourImg = edgeDetect(img.L)
	contourImg = (contourImg+10) / np.max(contourImg)
	for i in range(0, img.H):
		for j in range(0, img.W):
			nowIndex = i*img.W + j
			# if i-1>=0 : # now & up(nowIndex-W)
			# 	edge.append([nowIndex, nowIndex-img.W])
			# 	weight = computeWeight(i, j, i-1, j, contourImg)
			# 	edgeWeight.append(weight*pairwise)
			
			if i+1<img.H : # now & down(nowIndex+W)
				edge.append([nowIndex, nowIndex+img.W])
				weight = computeWeight(i, j, i+1, j, contourImg)
				edgeWeight.append(weight*pairwise)

			# if j-1>=0 : # now & left(nowIndex-1)
			# 	edge.append([nowIndex, nowIndex-1])
			# 	weight = computeWeight(i, j, i, j-1, contourImg)
			#	edgeWeight.append(weight*pairwise)

			if j+1<img.W : # now & right(nowIndex+1)
				edge.append([nowIndex, nowIndex+1])			
				weight = computeWeight(i, j, i, j+1, contourImg)
				edgeWeight.append(weight*pairwise)


	edge = np.array(edge).astype('int32')
	edgeWeight = np.array(edgeWeight).astype('int32')#.reshape(edge.shape[0],K,K)
	return edge, edgeWeight
				

def edgeDetect(img):
	sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobelSize)
	sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobelSize)
	out = np.zeros(np.shape(sobelX))
	out = np.sqrt(sobelX**2+sobelY**2)
	return out

def computeWeight(i1, j1, i2, j2, contourImg):
	return int((1./contourImg[i1,j1] + 1./contourImg[i2,j2]) / 2.)

def pairwiseCost(centroids):
	pairwise = np.zeros((K,K))
	for i in range(0, K):
		for j in range(0, K):
			pairwise[i,j] = np.linalg.norm(centroids[i]-centroids[j], 2)
	return pairwise.astype('int32')

def graphCut(img, centroids, unary):
	link = linkWeight(img)
	unary = (alpha*unary).astype('int32')
	pairwise = pairwiseCost(centroids)
	numIter = -1
	edge, edgeCost = edgeAndWeight(img, pairwise)

	# unary2D = np.zeros((img.H,img.W,K)).astype('int32')
	# for i in range(0,img.H):
	# 	for j in range(0,img.W):
	# 		unary2D[i,j,:] = unary[i*img.W+j,:]

	img.labels = pyqpbo.alpha_expansion_general_graph(edge,unary,edgeCost)
	# img.labels = pyqpbo.alpha_expansion_grid(unary2D, pairwise)
	# img.labels = pygco.cut_simple(unary2D, pairwise, n_iter=1)
	# img.labels = pygco.cut_simple_vh(unary2D, pairwise, costV, costH)
	# img.labels = pygco.cut_from_graph(link, unary, pairwise, n_iter=numIter, algorithm='swap')