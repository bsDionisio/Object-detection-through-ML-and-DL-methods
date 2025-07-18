import cv2
import globals
import numpy as np

class Kaze:
    def __init__(self):
        #Initialize the SIFT detector
        self.kaze = cv2.xfeatures2d.KAZE_create()
        self.grayFrame = None
        self.keypoints_left = None
        self.descriptors_right = None
        self.descriptors_left = None
        self.keypoints_right = None

    def find_key_points_logo(self, logo):
        # find the keypoints and descriptors with SIFT
        self.grayFrame = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        self.img1 = logo.copy()
        keypointsL = globals.detector.detect(self.img1, None)  #find the keypoints
        self.keypoints_left, self.descriptors_left = globals.descriptor.compute(self.grayFrame, keypointsL)

    def find_key_points_frame(self, frame):
        # find the keypoints and descriptors with SIFT
        self.grayLogo = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        keypointsR = globals.detector.detect(self.img1, None)  #find the keypoints
        self.keypoints_right, self.descriptors_right = globals.descriptor.compute(self.grayFrame, keypointsR)

    def find_matches(self, frame):
         # Create BFMatcher object
        BFMatcher = cv2.BFMatcher(normType = cv2.NORM_L2, crossCheck = True)

        # Matching descriptor vectors using Brute Force Matcher
        matches = BFMatcher.match(queryDescriptors = descriptors1, trainDescriptors = descriptors2)