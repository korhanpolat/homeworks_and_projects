#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:12:48 2018

train
134 --> image with text around
13 --> redscale


@author: korhan
"""


import cv2
import numpy as np
#from os import listdir
#from os.path import isfile, join
import matplotlib.pyplot as plt
#import data_augmentation as da
#from skimage.segmentation import chan_vese
from skimage import util, filters, color
from skimage.morphology import watershed
from skimage.filters import try_all_threshold
#from my_utils import load_obj

def plot_1D_hist(img, clr='b'):
    
    plt.figure()
    if len(img.shape) == 3:
        color = ('b','g','r')
        for i,col in enumerate(color):
            histr = cv2.calcHist([img],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
    else:
        histr = cv2.calcHist([img],[0],None,[256],[0,256])
        plt.plot(histr,color = clr)
    plt.xlim([0,256])
    plt.show()


def plot_2D_hist(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    plt.imshow(hist,interpolation = 'nearest')
    plt.show()

def stack_and_plot(img1,img2):
    res = np.hstack((img1,img2)) #stacking images side-by-side
    cv2.imshow('stacked image',res)    
        

def patch_wise_hist_eq(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(img)
    stack_and_plot(img,equ)
    return equ


def corner_detection_harris(img):
    if len(img.shape) == 3: gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else: 
        gray = img
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)
    
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    
    cv2.imshow('dst',img)


def corner_detection_shitomasi(img):
    if len(img.shape) == 3: gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else: 
        gray = img
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1)
    
    cv2.imshow('corners',img)


def sift_features(img):
    
    if len(img.shape) == 3: gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return img

def gen_sift_features(gray_img):
    """     kp is the keypoints
     desc is the SIFT descriptors, they're 128-dimensional vectors
     that we can use for our final features """
    sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def surf_features(img):
    if len(img.shape) == 3: img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(img,None)
    

def orb_detector(img):
    if len(img.shape) == 3: img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create()

    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(img2), plt.show()
    
    return kp,des

def my_watershed(img,compactness,n_seeds = 9):
    if len(img.shape) == 3: img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    edges = filters.sobel(img)
    grid = util.regular_grid(img.shape, n_points=n_seeds)
    
    seeds = np.zeros(img.shape, dtype=int)
    seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1
    
    w0 = watershed(edges, seeds, compactness=compactness)
    
    fig, ax = plt.subplots()
    ax.imshow(color.label2rgb(w0, img))
    ax.set_title('Compactness:' + str(compactness))

def draw_ORB_matches(img1,img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1,des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    plt.imshow(img3),plt.show()


def try_thresholdings(img):    
    """ try all thresholding """
    fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()

def read_img(path,gray=False):
    if gray:
        return cv2.imread(path,0)
    else:
        return cv2.imread(path)

""" smoothing """
def gaus_blur(img): return cv2.GaussianBlur(img,(5,5),0)

def resize_img(img,n_pixels):
    
    h,w = img.shape
    r = np.sqrt(n_pixels/float(h*w))
    
    img = cv2.resize(img,(int(r*w),int(r*h)))

    return img

def xy_dist(a,b,p=2):
    
    return (np.abs(a[0]-b[0])**2 + np.abs(a[1]-b[1])**p)**(0.5)
    

def feat_detector(img1,method='sift'):
    
    if method == 'sift': detector = cv2.xfeatures2d.SIFT_create()
    elif method == 'orb': detector = cv2.ORB_create()
    
    kp1, desc1 = detector.detectAndCompute(img1, None)
    return kp1,desc1    
    
def first_n_good_matches(kp1,desc1, kp2,desc2,n=10,threshold = 200,homography_test=True):
    """
        given keypoints and descriptors of two images, this function finds the matching
        descriptors, compares corresponding keypoints' spatial locations and returns 
        first n good mathces
    """
    MIN_MATCH_COUNT = 5
    ratio = 0.8
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1,desc2,k=2)
    # Apply ratio test
    good = []
    
    while len(good)<MIN_MATCH_COUNT and len(matches)>0:

        for m in matches:

            if len(m)==2:
                desc_d = [m[i].distance for i in range(2)]
                if desc_d[0] < ratio*desc_d[1]:
                            
                    xy = kp1[m[0].queryIdx].pt
                    kp_dist = [xy_dist(kp2[m[i].trainIdx].pt,xy,p=2.2) for i in range(2)]
                
                    if kp_dist[0] < threshold: good.append(m[0])
            else: return []
        ratio += 0.05
        threshold += 20
        
    good = sorted(good,key= lambda x: x.distance)
#    print 'len(good): ',len(good)
    
    if homography_test:
        
        # homography test
    
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        a = [good[i]  for i in range(len(good)) if matchesMask[i]==1]

    else: a = good    

    n = min(len(a),n)
    
#    if n< MIN_MATCH_COUNT: print "Not enough matches are found - %d/%d" % (len(a),MIN_MATCH_COUNT)
    print "matches found - %d" % (n)
    
    
    return a[0:n]



#
#    img1_matches = [ (kp1[m.queryIdx].pt,desc1[m.queryIdx]) for m in a[0:n] ]
#    img2_matches = [ (kp2[m.trainIdx].pt,desc2[m.trainIdx]) for m in a[0:n] ]
#
#    return img1_matches,img2_matches    
#    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
#
#    img1 = cv2.drawKeypoints(img1,[kp1[m.queryIdx] for m in good],img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#    img2 = cv2.drawKeypoints(img2,[kp2[m.trainIdx] for m in good],img2,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,flags=2)
#    plt.figure(), plt.imshow(img3), plt.show()
#    

    

#rootdir = '/home/korhan/.kaggle/competitions/whale-categorization-playground'
#trainDir = join(rootdir,'grup6')
#
#
#train_input = load_obj('train_grup7')
#uniq_ids = load_obj('uniq_ids_7')
#
#
#same_class_rows = train_input[train_input.Id == uniq_ids.Id[5]]
#same_class_rows = same_class_rows.reset_index(drop=True)
#
#img1 =  gaus_blur(resize_img(read_img(join(trainDir,same_class_rows.Image[0]),gray=True),2e5))
#img2 =  gaus_blur(resize_img(read_img(join(trainDir,same_class_rows.Image[2]),gray=True),2e5))
##img3 =  gaus_blur(resize_img(read_img(join(trainDir,same_class_rows.Image[4]),gray=True),2e5))
#
#p = 2.2
#threshold = 300
#
## knn based
#k = 2
#
#
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,a,None,flags=2)
#plt.figure(), plt.imshow(img3), plt.show()
#
#
#img4 = cv2.drawKeypoints(img1,[kp1[m.queryIdx] for m in a],img1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#plt.figure(), plt.imshow(img4), plt.show()
#
#
#


""" 
# flann based
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(desc1,desc2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in xrange(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        print m.queryIdx,n.trainIdx

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()

"""


# plain matches
# Initiate SIFT detector
#orb = cv2.ORB_create()
#
## find the keypoints and descriptors with SIFT
#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)
#
## create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
## Match descriptors.
#matches = bf.match(des1,des2)
#threshold = 150
#good_m = []
#for m in matches:
#    xy = kp1[m.queryIdx].pt
#    kp_dist = xy_dist(kp2[m.trainIdx].pt,xy,p) 
#    if kp_dist < threshold:
#        good_m.append(m)
#    
## Sort them in the order of their distance.
##matches = sorted(matches, key = lambda x:x.distance)
#matches = sorted(good_m, key = lambda x:x.distance)
#
#
#N_MATCHES = 50
#
#match_img = cv2.drawMatches(
#    img1, kp1,
#    img2, kp2,
#    matches[:N_MATCHES], img1.copy(), flags=0)
#
#plt.figure(figsize=(12,6))
#plt.imshow(match_img);
#
#














""" noise removal - filtering """

#median = cv2.medianBlur(img,5)
#blurbi = cv2.bilateralFilter(img,9,75,75)
#blurgaus = cv2.GaussianBlur(img,(5,5),0)
#
#cv2.imshow("median",median)
#cv2.imshow('blurbi',blurbi)
#cv2.imshow('blurgaus',blurgaus)




""" ORB """
#orb_detector(img)


""" SIFT """
#img_sift = sift_features(img)
#cv2.imshow('sift',img_sift)

""" corner detection """

#corner_detection_shitomasi(img)

""" histogram equalization """
#equ = cv2.equalizeHist(img)
#stack_and_plot(img,equ)

""" local patches """
#equ = patch_wise_hist_eq(img)

""" 2D hsv hist """
#plot_2D_hist(img)

""" plotting """


#plot_1D_hist(img)
#plot_1D_hist(equ,'r')




""" data augmentation """
#imgs = [da.augmentation_pipeline(img) for _ in range(5)]

    