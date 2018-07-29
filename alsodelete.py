import numpy as np
import cv2
from matplotlib import pyplot as plt

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread('/Users/andyyang/Desktop/Screen Shot 2a018-07-27 at 4.20.40 PM.png',0)          # queryImage
img2 = cv2.imread('/Users/andyyang/Desktop/cratess.png',0) # trainImage

#img1=img1[400:800,0:400]
h,w=img1.shape
img11=img1[0:int(h/2),0:w]
img12=img1[int(h/2):h,0:w]
arr=[]
for a in range(0,3):
    for b in range(0,3):
        arr.append(img1[int(a*h/3):int((a+1)*h/3),int(b*w/3):int((b+1)*w/3)])
print(arr[0])
MIN_MATCH_COUNT = 7
imgarray = []
kp=[]
realgood=[]

for a in range(0,9):
    img1=arr[a]
    # Initiate SIFT detector

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    print(des2)
    print(kp1)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    #print(matches)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
            print(m)

    #if len(good)>=4:
    realgood.extend(good)
    print('hi')
    print(realgood)
    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # matchesMask = mask.ravel().tolist()

    h,w = img2.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # dst = cv2.perspectiveTransform(pts,M)
    # img1 = cv2.polylines(img1,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    h,w=img1.shape
    for element in kp1:

       
        newpt=(element.pt[0]+int((a%3)*w),
        element.pt[1]+int((a-(a%3))*h/3))
        element.pt=newpt
    kp.extend(kp1)
    # else:
    #     print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    #     matchesMask = None

    
    # draw_params = dict(matchColor = None, # draw matches in green color
    #                    singlePointColor = None,
    #                    matchesMask = matchesMask, # draw only inliers
    #                    flags = 2)    
    #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)  
    
    h,w=img1.shape[0:2]
    

   # plt.imshow(img3, 'gray'),plt.show()
    imgarray.append(img1[0:int(h),0:int(w)])


row1=np.concatenate((imgarray[0], imgarray[1],imgarray[2]), axis=1)
row2=np.concatenate((imgarray[3], imgarray[4],imgarray[5]), axis=1)
row3=np.concatenate((imgarray[6], imgarray[7],imgarray[8]), axis=1)
vis = np.concatenate((row1,row2,row3),axis=0)
ee = cv2.drawKeypoints(vis,kp,None)
FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(kp,kp2,k=2)
# #print(matches)
# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.5*n.distance:
#         good.append(m)
#         print(m)
plt.imshow(ee,'gray')
plt.show()
vis = cv2.drawMatches(vis,kp,img2,kp2,realgood,None)  
plt.imshow(vis, 'gray'),plt.show()

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], np.array([]), (0, 0, 255),
                     #flags=2)