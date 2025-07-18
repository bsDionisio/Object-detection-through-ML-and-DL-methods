import cv2 
import numpy as np
from matplotlib import pyplot as plt

class Brief:
    def __init__(self):
        # Create the BRIEF extractor and compute the descriptors
        self.brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.keyPointsLogo = None
        self.keyPointsFrame = None
        self.descriptorsLogo = None
        self.descriptorsFrame = None
        self.img1 = None  # logo
        self.img2 = None  # frame

    def find_key_points_logo(self, logo):
        # Convert it to gray scale
        self.img1 = logo.copy()
        grayLogo = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)

        # Detect the CenSurE key points
        star = cv2.xfeatures2d.StarDetector_create()
        self.keyPointsLogo = star.detect(grayLogo, None)

    def find_key_points_frame(self, frame):
        # Convert it to gray scale
        self.img2 = frame.copy()
        grayFrame = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

        # Detect the CenSurE key points
        star = cv2.xfeatures2d.StarDetector_create()
        self.keyPointsFrame = star.detect(grayFrame, None)

    def find_matches(self, frame):
        # Compute descriptors for the logo and the frame
        self.keyPointsLogo, self.descriptorsLogo = self.brief.compute(self.img1, self.keyPointsLogo)
        self.keyPointsFrame, self.descriptorsFrame = self.brief.compute(self.img2, self.keyPointsFrame)

        # Check if descriptors were computed successfully
        if self.descriptorsFrame is None:
            print("Descriptors not computed. Make sure keypoints are detected properly.")
            return

        # Create BFMatcher to match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors between the logo and the frame
        matches = bf.match(self.descriptorsLogo, self.descriptorsFrame)

        # Sort the matches based on distance (best matches first)
        goodMatches = sorted(matches, key=lambda x: x.distance)


        # Draw keypoints for self.img2 (frame)
        result = cv2.drawMatches(self.img1, self.keyPointsLogo, self.img2, self.keyPointsFrame, 
                                 goodMatches, None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

        # Display the results
        cv2.imshow('Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

