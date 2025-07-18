# # BRIEF(Binary Robust Independent Elementary Features)

# ## Import resources and display image

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Brief:

    def __init__(self):
        #Initialize the SURF detector(Hessian Threshold is a parameter controlling keypoint detection)

        self.FAST = cv2.FastFeatureDetector_create() 
        self.BRIEF = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.grayFrame = None
        self.keypoints_left = None
        self.keypoints_left = None
        self.descriptors_left = None
        self.keypoints_right = None
        self.keypoints_without_size = None
        self.keypoints_with_size = None

    def find_key_points_logo(self, logo):

        # find the keypoints and descriptors with BRIEF
        self.grayFrame = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        self.img1 = logo.copy()
        fast_keypoints_frame = self.FAST.detect(self.grayFrame, None)
        self.keypoints_left, self.descriptors_left = self.BRIEF.compute(self.grayFrame, fast_keypoints_frame)

    def find_key_points_frame(self, frame):

        # find the keypoints and descriptors with BRIEF
        self.grayLogo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        fast_keypoints_logo = self.FAST.detect(self.grayLogo, None)
        self.keypoints_right, self.descriptors_right = self.BRIEF.compute(self.grayLogo, fast_keypoints_logo)

    def find_matches(self, frame):

        self.find_key_points_frame(frame)

        # ## Matching Keypoints
        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

        # Perform the matching between the BRIEF descriptors of the training image and the test image
        matches = bf.match(self.descriptors_left, self.descriptors_right)

        # The matches with shorter distance are the ones we want.
        matches = sorted(matches, key = lambda x : x.distance)

        result = cv2.drawMatches(self.img1, self.keypoints_left, self.img2, self.keypoints_right, matches, self.img2, flags = 2)

        # Print the number of keypoints detected in the sample image
        print("Number of Keypoints Detected In The Sample Image: ", len(self.keypoints_left))

        # Print the number of keypoints detected in the overall image
        print("Number of Keypoints Detected In The Overall Image: ", len(self.keypoints_right))

        # Print total number of matching points between the sample and overall images
        print("\nNumber of Matching Keypoints Between The Sample and Overall Images: ", len(matches))

        cv2.imwrite("brief_result.png", result)
        cv2.imshow('result', result)
        cv2.waitKey()
        cv2.destroyAllWindows()