import PIL
from PIL import ImageGrab
import numpy as np 
import cv2
import pyautogui
import time
from matplotlib import pyplot as plt

def findImage(queryimg,templateimg,TOLERANCE,MIN_MATCH_COUNT):
	
	sift = cv2.xfeatures2d.SIFT_create()
	MIN_MATCH_COUNT = 4
	h,w=queryimg.shape
	kp=[]
	realgood=[]

	kp1, des1 = sift.detectAndCompute(queryimg,None)
	kp2, des2 = sift.detectAndCompute(templateimg,None)
	#FLANN MATCHING
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	# Brute force matching?
	# bf = cv2.BFMatcher()
	# matches = bf.knnMatch(des1,des2, k=2)

	good = []
	for m,n in matches:
	    if m.distance < TOLERANCE*n.distance:
	        good.append(m)
	        #print(m)

	if len(good)>=1:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		matchesMask = mask.ravel().tolist()
		h,w = templateimg.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		print(M)
		dst = cv2.perspectiveTransform(pts,M)
		print(dst)
		queryimg = cv2.polylines(queryimg,[np.int32(dst)],True,255,3, cv2.LINE_AA)

	else:
	    matchesMask = None
	draw_params = dict(matchColor = None, 
	                   singlePointColor = None,
	                   matchesMask = matchesMask, 
	                   flags = 2)    
	h,w=queryimg.shape
	kp.extend(kp1)
	goodtemp=good
	for match in goodtemp:
	    match.queryIdx+=len(kp)-len(kp1)
	    realgood.extend(goodtemp)
	vis = cv2.drawMatches(queryimg,kp,templateimg,kp2,realgood,None)  
	return vis
# queryimg = cv2.imread('/Users/andyyang/Desktop/Screen Shot 2a018-07-27 at 4.20.40 PM.png',0)
# templateimg = cv2.imread('/Users/andyyang/Desktop/cratess.png',0)

plt.imshow(
	findImage(
		cv2.imread('Screen Shot 2018-07-27 at 12.30.53 PM.png',0),
		cv2.imread('/Users/andyyang/Desktop/cratess.png',0),
		0.7,
		4))
plt.show()

