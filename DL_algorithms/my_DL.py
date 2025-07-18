import cv2 as cv2
import numpy as np
import torch
from SuperPoint.superpoint_pytorch import SuperPoint
from SuperPoint.notebooks.utils import plot_imgs
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class DL:
    def __init__(self):
        self.sift = cv2.SIFT_create(contrastThreshold=0.09, edgeThreshold=10)
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
        detection_thresh = 0.005
        nms_radius = 5
        self.model = SuperPoint(detection_threshold=detection_thresh, nms_radius=nms_radius)
        self.model.load_state_dict(torch.load('SuperPoint/weights/superpoint_v6_from_tf.pth'))
        self.model.eval()

    def find_key_points_logo(self, frame):
        # find the keypoints and descriptors with SIFT
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img1 = frame.copy()

        # using model
        image = self.img1.mean(-1) / 255
        image = np.pad(image, [(0, int(np.ceil(s / 8)) * 8 - s) for s in image.shape[:2]])
        with torch.no_grad():
            pred_th = self.model({'image': torch.from_numpy(image[None, None]).float()})


        # points_th = pred_th['keypoints'][0]

        #plot_imgs([image], cmap='gray', titles=[f'PyTorch model, {len(points_th)} points'])
        #plt.scatter(*points_th.T, lw=0, s=4, c='lime')
        #plt.show()

        self.keypoints_left = pred_th['keypoints'][0].numpy()
        self.descriptors_left = pred_th['descriptors'][0].numpy()

        # self.keypoints_left, self.descriptors_left = self.sift.detectAndCompute(gray, None)

    def find_key_points_frame(self, frame):
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.img2 = frame.copy()
        # self.keypoints_right, self.descriptors_right = self.sift.detectAndCompute(gray, None)

        image = self.img2.mean(-1) / 255
        image = np.pad(image, [(0, int(np.ceil(s / 8)) * 8 - s) for s in image.shape[:2]])
        with torch.no_grad():
            pred_th = self.model({'image': torch.from_numpy(image[None, None]).float()})

        self.keypoints_right = pred_th['keypoints'][0].numpy()
        self.descriptors_right = pred_th['descriptors'][0].numpy()

    def find_matches(self, frame):
        self.find_key_points_frame(frame)

        # knn_matches = self.matcher.knnMatch(self.descriptors_left, self.descriptors_right, 2)

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(self.descriptors_right)
        distances, indices = nbrs.kneighbors(self.descriptors_left)

        good_matches = []
        for i, (distance, index) in enumerate(zip(distances, indices)):
            if distance[0] < 0.75 * distance[1]:  # Usando o teste de razão de Lowe
                good_matches.append(cv2.DMatch(_distance=distance[0], _imgIdx=0, _queryIdx=i, _trainIdx=index[0]))

        # -- Draw matches
        keypoints1_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in self.keypoints_left]
        keypoints2_cv = [cv2.KeyPoint(x=kp[0], y=kp[1], size=1) for kp in self.keypoints_right]
        img_matches = cv2.drawMatches(self.img1, keypoints1_cv, self.img2, keypoints2_cv, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('matchesResult', img_matches)

        # -- Localize the object
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)

        rows, cols, _ = self.img2.shape

        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches

            if keypoints2_cv[good_matches[i].trainIdx].pt[0] > cols - 1 or \
                    keypoints2_cv[good_matches[i].trainIdx].pt[1] > rows - 1 or \
                    keypoints2_cv[good_matches[i].trainIdx].pt[0] < 0 or \
                    keypoints2_cv[good_matches[i].trainIdx].pt[1] < 0:
                cv2.imshow('frame1', self.img2)
                continue

            obj[i, 0] = keypoints1_cv[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints1_cv[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints2_cv[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints2_cv[good_matches[i].trainIdx].pt[1]

        if len(obj) < 4:
            cv2.imshow('frame2', self.img2)
            return

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
            cv2.imshow('frame3', self.img2)
            return

        cv2.line(self.img2, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])),
                 (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img2, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])),
                 (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img2, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])),
                 (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
        cv2.line(self.img2, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])),
                 (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)

