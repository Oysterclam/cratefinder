import numpy as np
import cv2
from matplotlib import pyplot as plt

sift = cv2.xfeatures2d.SIFT_create()

queryimg = cv2.imread('/Users/andyyang/Desktop/Screen Shot 2018-07-29 at 10.39.08 AM.png',0)
templateimg = cv2.imread('/Users/andyyang/Desktop/Screen Shot 2018-07-30 at 4.12.30 PM.png',0)
TOLERANCE=0.5



#queryimg=queryimg[400:800,0:400]
h,w=queryimg.shape
queryimg1=queryimg[0:int(h/2),0:w]
querytemplateimg=queryimg[int(h/2):h,0:w] 
arr=[]
for a in range(0,3):
    for b in range(0,3):
        arr.append(queryimg[int(a*h/3):int((a+1)*h/3),int(b*w/3):int((b+1)*w/3)])
MIN_MATCH_COUNT = 4
imgarray = []
kp=[]

realgood=[]

for a in range(0,9):
    queryimg=arr[a]
 
    kp1, des1 = sift.detectAndCompute(queryimg,None)
    kp2, des2 = sift.detectAndCompute(templateimg,None)
    #FLANN MATCHING
    FLANN_INDEX_KDTREE = 1
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)
        #print(matches)
    #BRUTE FORCE MATCHING
    #bf = cv2.BFMatcher()
    # matches = bf.knnMatch(des1,des2, k=2)
    good = []
    for m,n in matches:
        if m.distance < TOLERANCE*n.distance:
            good.append(m)
            #print(m)

    if len(good)>=1:
        #print(len(good))
       
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        # h,w = templateimg.shape
        # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        # dst = cv2.perspectiveTransform(pts,M)
        # queryimg = cv2.polylines(queryimg,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    else:
        #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    
    draw_params = dict(matchColor = None, 
                       singlePointColor = None,
                       matchesMask = matchesMask, 
                       flags = 2)    
    #img3 = cv2.drawMatches(queryimg,kp1,templateimg,kp2,good,None,**draw_params)  
 
    h,w=queryimg.shape
    for element in kp1:

       
        newpt=(element.pt[0]+int((a%3)*w),
        element.pt[1]+int((a-(a%3))*h/3))

        element.pt=newpt
      
    kp.extend(kp1)
    goodtemp=good
    for match in goodtemp:
        match.queryIdx+=len(kp)-len(kp1)

    realgood.extend(goodtemp)

        
  
  
    #plt.imshow(img3, 'gray'),plt.show()
    h,w=queryimg.shape[0:2]
    imgarray.append(queryimg[0:int(h),0:int(w)])


row1=np.concatenate((imgarray[0], imgarray[1],imgarray[2]), axis=1)
row2=np.concatenate((imgarray[3], imgarray[4],imgarray[5]), axis=1)
row3=np.concatenate((imgarray[6], imgarray[7],imgarray[8]), axis=1)   
vis = np.concatenate((row1,row2,row3),axis=0)
ee = cv2.drawKeypoints(vis,kp,None)

# for match in realgood:
#     print(match.queryIdx)
#plt.imshow(ee,'gray')
#plt.show()
vis = cv2.drawMatches(vis,kp,templateimg,kp2,realgood,None)  
plt.imshow(vis, 'gray'),plt.show()

#img3 = cv2.drawMatches(queryimg,kp1,templateimg,kp2,matches[:10], np.array([]), (0, 0, 255),
                     #flags=2)
