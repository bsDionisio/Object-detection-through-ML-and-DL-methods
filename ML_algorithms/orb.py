import cv2 as cv2
import numpy as np


class Orb:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=3000,         # More features
        scaleFactor=1.1,        # Finer scale pyramid
        nlevels=8,              # More pyramid levels
        edgeThreshold=15,       # Smaller: detect closer to edges
        patchSize=3)            # Larger patch = more robust descriptors
        self.keypoints_left = None
        self.descriptors_left = None
        self.keypoints_right = None
        self.descriptors_right = None
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.img1 = None  # logo
        self.img2 = None  # frame

    def find_key_points_logo(self, frame):
        # find the keypoints and descriptors with SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img1 = frame.copy()
        self.keypoints_left, self.descriptors_left = self.orb.detectAndCompute(gray, None)

    def find_key_points_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        self.keypoints_right, self.descriptors_right = self.orb.detectAndCompute(gray, None)

    def find_matches(self, frame):
        self.find_key_points_frame(frame)

        matches = self.matcher.match(self.descriptors_left, self.descriptors_right)

        good_matches = sorted(matches, key=lambda x: x.distance)
        
        # -- Draw matches
        img_matches = np.empty(
            (max(self.img1.shape[0], self.img2.shape[0]), self.img1.shape[1] + self.img2.shape[1], 3),
            dtype=np.uint8)
        cv2.drawMatches(self.img1, self.keypoints_left, self.img2, self.keypoints_right, good_matches, img_matches,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        
        # Print the number of keypoints detected in the sample image
        print("Number of Keypoints Detected In The Sample Image: ", len(self.keypoints_left))
        # Print the number of keypoints detected in the overall image
        print("Number of Keypoints Detected In The Overall Image: ", len(self.keypoints_right))
        # Print total number of matching points between the sample and overall images
        print("\nNumber of Matching Keypoints Between The Sample and Overall Images: ", len(matches))
        cv2.imwrite("orb_result.png", img_matches)
        cv2.imshow('matches', img_matches)