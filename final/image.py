import cv2
import numpy as np
from parameters import *

class Image:
	def __init__(self,filename):
		self.bgr = cv2.imread(filename)
		self.Lab = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2Lab)
		self.L = self.Lab[:,:,0]
		self.Lg1 = cv2.GaussianBlur(self.L, (0,0), blurSig1)
		self.Lg2 = cv2.GaussianBlur(self.L, (0,0), blurSig2)
		self.H, self.W = self.L.shape
		self.pixel_num = (self.W-2*n)*(self.H-2*n)
		a = self.Lab[n:self.H-n,n:self.W-n,1].reshape((self.pixel_num,1))
		b = self.Lab[n:self.H-n,n:self.W-n,2].reshape((self.pixel_num,1))
		self.kmeansfeatures = np.concatenate((a,b),axis = 1)
		self.labels = []
		self.features = []
