import cv2
import random
import numpy as np
from Model import Model, DecoderType
import tensorflow as tf
#tf.disable_v2_behavior() 


# from DataLoader import DataLoader, Batch
fnCharList = '../model/charList.txt'
decoderType = DecoderType.BestPath
model = Model(open(fnCharList).read(),  mustRestore=True)




class Batch:
	"batch containing images and ground truth texts"
	def __init__(self, gtTexts, imgs):
		self.imgs = np.stack(imgs, axis=0)
		self.gtTexts = gtTexts

def preprocess(img, imgSize, dataAugmentation=False):
	"put img into target img of size imgSize, transpose for TF and normalize gray-values"

	# there are damaged files in IAM dataset - just use black image instead
	if img is None:
		img = np.zeros([imgSize[1], imgSize[0]])

	# increase dataset size by applying random stretches to the images
	if dataAugmentation:
		stretch = (random.random() - 0.5) # -0.5 .. +0.5
		wStretched = max(int(img.shape[1] * (1 + stretch)), 1) # random width, but at least 1
		img = cv2.resize(img, (wStretched, img.shape[0])) # stretch horizontally by factor 0.5 .. 1.5
	
	# create target image and copy sample image into it
	(wt, ht) = imgSize
	(h, w) = img.shape
	fx = w / wt
	fy = h / ht
	f = max(fx, fy)
	newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
	img = np.array(img, dtype='uint8')
	img = cv2.resize(img, newSize)
	target = np.ones([ht, wt]) * 255
	target[0:newSize[1], 0:newSize[0]] = img

	# transpose for TF
	img = cv2.transpose(target)

	# normalize
	(m, s) = cv2.meanStdDev(img)
	m = m[0][0]
	s = s[0][0]
	img = img - m
	img = img / s if s>0 else img
	return img

def infer(Img):
	"recognize text in image provided by file path"
	# print(Img.shape)
	for i in range(len(Img)):
		Img[i] = preprocess(Img[i], Model.imgSize)
		# cv2.imwrite("image3.jpg",Img[i])
	batch = Batch(None, Img)
	return model.inferBatch(batch, False )


