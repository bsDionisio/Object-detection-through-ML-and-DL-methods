from __future__ import print_function
import cv2 as cv2
import numpy as np
import argparse
from math import sqrt

class Akaze:
    def __init__(self):
        self.akaze = cv2.AKAZE_create()  #(descriptor_size=0, descriptor_channels=3, threshold=0.001)

        self.keypoints_left = None
        self.descriptors_left = None
        self.keypoints_right = None
        self.descriptors_right = None
        self.img1 = cv2.imread('data/logo.png')
        self.img2 = cv2.imread('data/frame.png')

    def find_matches(self, frame):

        parser = argparse.ArgumentParser(description='Code for AKAZE local features matching tutorial.')
        parser.add_argument('--input1', help='Path to input image 1.', default='data/frame.png')
        parser.add_argument('--input2', help='Path to input image 2.', default='data/logo.png')
        parser.add_argument('--homography', help='Path to the homography matrix.', default='data/H1to3p.xml')
        args = parser.parse_args()

        fs = cv2.FileStorage(cv2.samples.findFile('/home/bsdionisio/Desktop/teseBea/data/H1to3p.xml'), cv2.FILE_STORAGE_READ)
        homography = fs.getFirstTopLevelNode().mat()

        self.keypoints_left, self.descriptors_left = self.akaze.detectAndCompute(self.img1, None)
        self.keypoints_right, self.descriptors_right = self.akaze.detectAndCompute(self.img2, None)


        # Match the features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(self.descriptors_left, self.descriptors_right, k=2)    # typo fixed

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append([m])

        # cv2.drawMatchesKnn expects list of lists as matches.
        result = cv2.drawMatchesKnn(self.img1, self.keypoints_left, self.img2, self.keypoints_right, good, None, flags=2)
        cv2.imwrite("results/akaze_result.png", result)
        
        print('A-KAZE Matching Results')
        print('*******************************')
        # Print the number of keypoints detected in the training image
        print("Number of Keypoints Detected In The Sample Image: ", len(self.keypoints_left))
        # Print the number of keypoints detected in the query image
        print("Number of Keypoints Detected In The Overall Image: ", len(self.keypoints_right))
        # Print total number of matching points between the training and query images
        print("\nNumber of Matching Keypoints Between The Sample and Overall Images: ", len(good))

        cv2.imshow('Akaze', result)
        cv2.waitKey()
        cv2.destroyAllWindows()