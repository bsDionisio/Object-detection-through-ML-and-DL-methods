import cv2 as cv2
import numpy as np
import time


class SiftLight:
    def __init__(self):
        # self.sift = cv2.SIFT_create(contrastThreshold=0.09, edgeThreshold=10)
        self.sift = cv2.SIFT_create()
        self.keypoints_left = None
        self.descriptors_left = None
        self.keypoints_right = None
        self.descriptors_right = None
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # ou passar empty dictionary
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # self.matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        self.img1 = None  # logo
        self.img2 = None  # frame

    def find_key_points_logo(self, frame):
        # find the keypoints and descriptors with SIFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img1 = frame.copy()
        self.keypoints_left, self.descriptors_left = self.sift.detectAndCompute(gray, None)

    def find_key_points_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        self.keypoints_right, self.descriptors_right = self.sift.detectAndCompute(gray, None)

    def find_matches(self, frame):

        # calculate time
        start = time.time()
        self.find_key_points_frame(frame)
        #print(f"Time to find key points: {time.time() - start}")

        # -- Match descriptors
        start = time.time()
        knn_matches = self.matcher.knnMatch(self.descriptors_left, self.descriptors_right, 2)
        #print(f"Time to match descriptors: {time.time() - start}")

        # -- Filter matches using the Lowe's ratio test

        start = time.time()
        ratio_thresh = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

        #print(f"Time to filter matches: {time.time() - start}")

        # # -- Draw matches
        # img_matches = np.empty((max(self.img1.shape[0], self.img2.shape[0]), self.img1.shape[1] + self.img2.shape[1], 3),
        #                        dtype=np.uint8)
        # cv2.drawMatches(self.img1, self.keypoints_left, self.img2, self.keypoints_right, good_matches, img_matches,
        #                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow('matches', img_matches)

        # -- Localize the object
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)

        rows, cols, _ = self.img2.shape

        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches

            if self.keypoints_right[good_matches[i].trainIdx].pt[0] > cols - 1 or \
                    self.keypoints_right[good_matches[i].trainIdx].pt[1] > rows - 1 or \
                    self.keypoints_right[good_matches[i].trainIdx].pt[0] < 0 or \
                    self.keypoints_right[good_matches[i].trainIdx].pt[1] < 0:
                #cv2.imshow('result', self.img2)
                continue

            obj[i, 0] = self.keypoints_left[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_left[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = self.keypoints_right[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = self.keypoints_right[good_matches[i].trainIdx].pt[1]


        if len(obj) < 4:
            #cv2.imshow('result', self.img2)
            return None

        #cv2.imshow('result', self.img2)
        H, _ = cv2.findHomography(obj, scene, cv2.RANSAC)

        # -- Get the corners from the image_1 ( the object to be "detected" )
        obj_corners = np.empty((4, 1, 2), dtype=np.float32)
        obj_corners[0, 0, 0] = 0
        obj_corners[0, 0, 1] = 0
        obj_corners[1, 0, 0] = self.img1.shape[1]
        obj_corners[1, 0, 1] = 0
        obj_corners[2, 0, 0] = self.img1.shape[1]
        obj_corners[2, 0, 1] = self.img1.shape[0]
        obj_corners[3, 0, 0] = 0
        obj_corners[3, 0, 1] = self.img1.shape[0]

        try:
            scene_corners = cv2.perspectiveTransform(obj_corners, H)
        except:
            # cv2.imshow('result', self.img2)
            return None

        return scene_corners



        #cv2.imshow('result', self.img2)
