# # SIFT (Scale-Invariant Feature Transform)
# ## Import resources and display image

import cv2
import matplotlib.pyplot as plt
import numpy as np

class Sift:
    def __init__(self):
        #Initialize the SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.grayFrame = None
        self.keypoints_left = None
        self.descriptors_right = None
        self.descriptors_left = None
        self.keypoints_right = None

    def find_key_points_logo(self, logo):
        # find the keypoints and descriptors with SIFT
        self.grayFrame = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        self.img1 = logo.copy()
        self.keypoints_left, self.descriptors_left = self.sift.detectAndCompute(self.grayFrame, None)

    def find_key_points_frame(self, frame):
        # find the keypoints and descriptors with SIFT
        self.grayLogo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        self.keypoints_right, self.descriptors_right = self.sift.detectAndCompute(self.grayLogo, None)

    def find_matches(self, frame):

        # ## Matching Keypoints
        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

        # Perform the matching between the SIFT descriptors of the training image and the test image
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

        # Display the best matching points
        cv2.imwrite("sift_result.png", result)
        cv2.imshow('SIFT', result)
        cv2.waitKey()
        cv2.destroyAllWindows()
