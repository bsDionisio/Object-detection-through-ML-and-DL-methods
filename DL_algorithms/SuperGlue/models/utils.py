# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    #Resets start and last recorded times and will_print flags to False for all tracked names
    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    #Computes time delta dt since last_time. Applies exponential smoothing to update the running time for the given name
    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        #Updates self.times[name] with smoothed value.
        self.times[name] = dt
        #Sets self.will_print[name]=True to indicate this should be printed next
        self.will_print[name] = True
        #Updates self.lat_time to now
        self.last_time = now

    #Prints all timing values where will_print[name] is True
    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            #Shows each named section and its smoothed time
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        #Shows total time and FPS (frames per second)
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        #Depending on mewline, prints on a new line or overwrites current line
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = False
        self.video_file = False  #MODIFY WHEN VIDEO
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        
        #Creates a cv2.VideoCapture object from the camera index
        #Populates self.listing as a range (acts like frame indices)
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        #Starts an IP camera stream
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            #Use of a dummy listing range
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            #Uses Path.glob() with each pattern in image_glob to get matching image files
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            #Sorts and skips files based on self.skip
            self.listing.sort()
            #You probably don't want to skip anything if comparing two images
            #self.skip = 1
            self.listing = self.listing[::self.skip]
            #Truncates to max_length
            self.max_length = np.min([self.max_length, len(self.listing)])
            #Limit to only 2 images
            #self.max_length = min(2, len(self.listing))
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        #Assumes it's a video file
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            #Grabs total frame count
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            #Sets up listing as range of frame indices, aaplying skipping and limiting to max_length
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        #If the input is a camera (USB or IP) but OpenCV can't open it, an error is raised
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        #Checks if cv2.imread() failed to load the image
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        #Gets the originalwidth and height of the image
        w, h = grayim.shape[1], grayim.shape[0]
        #This function computes the new dimensions w_nem and h_new based on the original size 
        #and a target resize parameter stored in self.resize
        w_new, h_new = process_resize(w, h, self.resize)
        #Returns the processed image, now in grayscale and resized to the desired size
        #Output is a 2D NumPy array (height x weight), with pixel values in uint8 (range 0-255)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter. 
            Handles various input sources
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """


        #self.i is the current frame index; if i equals max_length, all frames are processed - 
        #so return None and False (end of stream)
        if self.i == self.max_length:
            return (None, False)
        #Checks if camera is enabled(i.e., we're reading from a wwebcam or video capture device)
        if self.camera:
            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                #When finished, ret indicates success, image is a copy of the latest IP camera image
                ret, image = self._ip_grabbed, self._ip_image.copy()
                #if ret==False, the camera stream failed -> mark _ip_running = False
                if ret is False:
                    self._ip_running = False
            #Regular camera input (e.g, webcam) - Reads a frame from a standard camera or video file using OpenCV's VideoCapture
            else:
                ret, image = self.cap.read()
            #If no frame was read successfully (ret==False), print an error and return failure
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            #Get original width and height of the frame
            w, h = image.shape[1], image.shape[0]
            #If reading from a video file, use .set() to jump to a specific frame index stored in self.listing[self.i]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            #Compute target size
            w_new, h_new = process_resize(w, h, self.resize)
            #Resize the image using OpenCV with a chosen interpolation method (self.interp)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            #Converts the resized image to grayscale
            #Assumes the image is in RGB format and converts to 1-channel
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #if self.camera is False, this means you're reading images from a list of files
        #Grabs the file path from self.listing and loads it using self.load_image()
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        #Increments the internal frame counter
        self.i = self.i + 1
        #Returns the image and True indicating success
        return (image, True)

    #Starts a background thread that continuously pulls frames from an IP camera stream, 
    # without blocking the main program
    def start_ip_camera_thread(self):
        #Creation of a new thread
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        #Starts the thread
        self._ip_thread.start()
        self._ip_exited = False
        return self

    #Runs in a background thread and continuoulsy reads frames from an IP camera (or any cv2.VideoCapture source)
    #It updates shared variables that other parts of the program can access safely
    def update_ip_camera(self):
        while self._ip_running:
            #Attempts to read a frame from the video capture object (self.cap), 
            # which is likely an IP camera stream or video source; ret is True if successful; img is the actual frame
            ret, img = self.cap.read()
            #If it fails
            if ret is False:
                #Stops running
                self._ip_running = False
                #Flags that the thread has exited
                self._ip_exited = True
                #Resets it to indicate failure to grab a frame
                self._ip_grabbed = False
                return

            #Saves the latest successfully read frame into a shared variable, so other treads can access the current frame
            self._ip_image = img
            #Sets a flag to indicate that a frame has been successfully grabbed and is ready to be read
            self._ip_grabbed = ret
            #Increments an internal counter that tracks how many frames have been grabbed so far
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    #Simple method to stop the background thread by setting it to False
    #This will cause the while loop in update_ip_camera() to exit on the next iteration
    #Should be called during shutdown or when you're done with the IP camera
    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

#Defines a function to compute the target dimensions for an image resize
def process_resize(w, h, resize):
    #Checks that the resize is either 1 element=meaning scale by largest dimension or "no resize"
    #2 elements=meaning explicitly set new width and height
    #Throws an error if the format is invalid
    assert(len(resize) > 0 and len(resize) <= 2)
    #If the resize has 1 numbeer greater than -1, that number is treated as a target size for the largest dimension
    if len(resize) == 1 and resize[0] > -1:
        #Computes a scale factor so the largest side becomes resize[0], preserving the aspect ratio
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    #If resize is [-1], no resizing - return original width and height
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    #If resize has exactly two values, they are interpreted as (target_width, target_height) directly
    #Aspect ratio is not preserved here
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    #Returns the computed target width and height
    return w_new, h_new


#Tiny helper that converts an image frame (NumPy array) into a PyTorch tensor ready for a model
def frame2tensor(frame, device):
    return torch.from_numpy(frame/255.).float()[None, None].to(device)

#Preprocessing helper that loads an image, resizes it, optionally rotates it, and prepares 
#it both as a NumPy array and as a PyTorch tensor
def read_image(path, device, resize, rotation, resize_float):
    #Reads the image in grayscale (1 channel) using OpenCV
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    #Shape is (height, width) so shape[1]=width, shape[0]=height
    w, h = image.shape[1], image.shape[0]
    #Uses earlier process_resize function to compute new dimensions
    w_new, h_new = process_resize(w, h, resize)
    #Computes scales=(width_scale, height_scale) - useful for later mapping results back to original size
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        #Converts to float first, then resize (slightly better precision)
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        #Resize first (in uint8), then convert to float
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        #Rotates the image 90º * rotation times
        image = np.rot90(image, k=rotation)
        #If the rotation is odd (90º or 270º), the width and height swap, so scale is reversed
        if rotation % 2:
            scales = scales[::-1]

    #Uses frame2Tensor to Normalize to 0-1, Add batch & channel dimensions and send to device
    inp = frame2tensor(image, device)
    #image->NumPy array (float32) for CPU-side processing/visualization; inp->PyTorch tensor for model input;
    #scales->scaling factors for later coordinate mapping
    return image, inp, scales


# --- GEOMETRY ---

#Estimates the relative camera pose (rotation R and translation T) between two views using matched keypoints
#Standard computer vision process that uses OpenCV's essential matrix tools
def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    #Check if enough matches; essential matrix estimation needs at least 5 points; if fewer, returns none
    if len(kpts0) < 5:
        return None
    
    #Average focal length between both cameras
    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    #Threshold normalized by focal length - converts pixel-based threshold to normalized camera units
    norm_thresh = thresh / f_mean

    #Converts from pixel coordinates to normalized image coordinates; uses broadcasting: subtract principal
    #point (K[0,2],K[1,2]), divide by focal length (K[0,0],K[1,1])
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    #Input: normalized points and identity intrinsics (np.eye(3)) since they're already normalized
    #threshold->RANSAC inlier distance;prob->desired probability of finding the correct set of inliers;method->robust estimation
    #Output:E->essential matrix(or multiple candidates stacked vertically);mask->boolean mask of inlier matches
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    #Ensure estimation succeeded
    assert E is not None

    best_num_inliers = 0
    ret = None
    #Keep track of the best (most inliers) solution
    for _E in np.split(E, len(E) / 3):
        #cv2.recoverPose extracts R and t from an essential matrix; n=number of inliers for this pose
        n, R, t, _ = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        #If this candidate has more inliers than the best so far
        if n > best_num_inliers:
            best_num_inliers = n
            #Save (R, t, inlier_mask) as ret
            ret = (R, t[:, 0], mask.ravel() > 0)
    #Returns R:rotation matrix(3x3), t:translation vector (3,), inliers:boolean mask array
    return ret


#It updates a camera intrinsic matrix K to match the image's new coordinate system after rotation
def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    #Ensures rotation is between 0º and 270º (in 90º steps)
    assert rot <= 3
    #If the rotation is 90º or 270º, the width and height swap; otherwise, stays the same
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    #fx,fy -> focal lengths in pixels; cx,cy -> principal point (optical center in pixels)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    #Handles cases like rot = 4 (full rotation) -> rot = 0
    rot = rot % 4
    #Rotation 90º
    if rot == 1:
        #Swap fx and fy (axes swap); new cx = old cy; new cy= w-1-old_cx (mirrored horizontally in new coordinate frame)
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    #Rotation 180º
    elif rot == 2:
        #No swap in fx and fy
        #Principal point moves to opposite corner: (w-1-cx, h-1-cy)
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3, 270º:
        #Swap fx and fy again; new cx=h-1-cy (mirrored vertically); new cy=old cx
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


#Used to rotate a 3D camera pose around the Z-axis (inplane rotation) by a fixed number of degrees (0º, 270º, 180º, or 90º)
def rotate_pose_inplane(i_T_w, rot):
    #Converts degrees into radians; Then, for each angle r, make a 4x4 homogeneous transformation matrix;
    #This gives 4 possible rotation matrices, stored in a list: 0=no rotation, 1=270º rotation, 2=180º rotation, 3=90º rotation
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    #Returns the result; This changes the orientation of the camera but keeps its world position
    return np.dot(rotation_matrices[rot], i_T_w)

#Adjusts the camera intrinsics matrix k when an image has been resized
def scale_intrinsics(K, scales):
    #Creates a 3x3 diagonal matrix; the inverse scaling is used because if the image in shrunk by a factor of 2, 
    # the focal length in pixel units should be halved
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)

#Tiny helper for converting 2D or 3D points into homogeneous coordinates
def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)

#Computes the symmetric epipolar distance between two sets of matched keypoints given the relative pose and camera intrinsics
def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    #This converts pixel coordinates -> normalized camera coordinates (until focal length, principal point at origin)
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    #Adds a 1 as the last coordinate so points become 3D vectors (x, y, 1)
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    #Build the skew-symmetric matrix for translation
    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    #Compute the Essential matrix
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


#Computes the angular difference (in degrees) between two rotation matrices
def angle_error_mat(R1, R2):
    #Compute the cosine of the rotation angle
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    #Prevents numerical instability
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    #Converts to degrees
    return np.rad2deg(np.abs(np.arccos(cos)))

#Computes the angle (in degrees) between two vectors v1 and v2 in 2D or 3D space
def angle_error_vec(v1, v2):
    #Compute normalization factor - This will be used to normalize the dot product, because the cosine formula requires unit vectors
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


#Measures how differentan an estimated camera pose is from the ground truth pose; computes rotation error and translation error
def compute_pose_error(T_0to1, R, t):
    #Top-left 3x3 part of T_0to1, the ground-truth rotation
    R_gt = T_0to1[:3, :3]
    #Last column (first 3 entries), the ground-truth translation
    t_gt = T_0to1[:3, 3]
    #Essential matrix decomposition (from epipolar geometry) gives translation up to sign -> t or -t are both valid; 
    # this introduces a 180º ambiguity
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    #Returns the regular difference (in degrees) between estimated rotation R and ground-truth rotation R_gt
    error_R = angle_error_mat(R, R_gt)
    #error_t: translation direction error (in degrees); error_R: rotation error (in degrees)
    return error_t, error_R


#This function is about computing the Area Under the Curve (AUC) of a pose estimation accuracy curve
def pose_auc(errors, thresholds):
    #Sorts errors in ascending order; sorting is necessary to compute a monotonic recall curve
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    #Recall here = fracton of samples below a certain error threshold
    recall = (np.arange(len(errors)) + 1) / len(errors)
    #Ensures the curve starts from (0 error, 0 recall)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    #For each threshld t (e.g. 5º, 10º, 20º), compute AUC of recall vs. error curve up to that threshold
    aucs = []
    for t in thresholds:
        #np.searchsorted finds where threshold t fits in the sorted errors
        last_index = np.searchsorted(errors, t)
        #Take recall/error values up to threshold; extend curve to exactly t by repeating last recall value at error t
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        #np.trapz(r,x=e)=trapezoidal integration under recall-vs-error curve; divide by t to normalize AUC (so it lies between 0 and 1)
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


#Simple visualization utility for plotting two images side by side in a clean, minimal way
def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    #Ensures exactly two imges are provided; this function is specifically for displaying image pairs (e.g., for comparison)
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    #figsize determines how large the figure will be
    figsize = (size*n, size*3/4) if size is not None else None
    #Creates 1xn subplot grid (so two panels side by side); an is an array of subplot axes
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    #Plots each image in grayscale (cmap='gray); forces pixel values between 0-255 for consistent contrast
    for i in range(n):
        #Removes tick marks on both axes -> clean look
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        #Removes the rectangular borders (spines) around each subplot
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    #Automatically adjusts subplot spacing; pad controls padding between plots
    plt.tight_layout(pad=pad)


#Visualization helper for plotting keypoints on top of the two images shown with plot_image_pair
def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    #plt.gcf() -> "Get Current Figure"; .axes-> list of all aces (subplots) in that figure; Since plot_image_pair 
    # creates 2 subplots, ax[0] is the first image, ax[1] the second
    ax = plt.gcf().axes
    #Plots keypoints on the first subplot; scatter places points at (x,y) locations from kpts0; c=colour -> point colour; 
    # s=ps -> marker size
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    #Same as above, but for second image; uses keypoints from kpts1
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


#Matches visualization; it takes two sets of corresponding keypoints and draws lines connecting them acress 
# two subplots (the side-by-side images)
#kpts0: keypoints in the first image; kpts1:corresponding keypoints in the second image; color: list/array of colors for 
# each match (e.g., all white "w" or per-match colours); lw:line width for the match connections; ps:point size for the keypoints
def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    #Gets the current figure (where images were drawn by plot_image_pair)
    fig = plt.gcf()
    #list of subplots (left=ax[0], right=ax[1])
    ax = fig.axes
    #Ensures the figure is rendered so transformations are ready
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    #First image keypoints in figure coordinates
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    #Second image keypoints in figure coordinates
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    #For each pair of keypoints i: draw a line between (fkpts0[i], fkpts1[i]); use fig.transFigure so the line spans both 
    # subplots correctly; c=color[i] allows each match to have its own color; lw sets line thickness; Stores them in fig.lines, 
    # which matplotlib automatically renders
    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    #Overlay keypoints on both images; same color array (so lines+points have matching colors); point size = ps
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


#This function ties everything together into a polished visualization of matching keypoints. It's the high-level function
#used to create the final match plots
def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, show_keypoints=False,
                       fast_viz=False, opencv_display=False,
                       opencv_title='matches', small_text=[]):

    #It calls make_matching_plot_fast (likely a faster, OpenCV-based method for real-time use); 
    # otherwise, continue with the matplotlib plotting pipeline
    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title, small_text)
        return

    #Uses the helper we explained earlier -> two subplots, one for each image
    plot_image_pair([image0, image1])
    #Plots all keypoints in both images; black circles first ('k' with size 4), then white dots on top ('w' with size 2);
    #This creates a nice outlined keypoint style (black border with white dot)
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    #Uses the prior helper -> draws lines between matched keypoints; color can be a per-match array (e.g., inliers=green, outliers=red)
    plot_matches(mkpts0, mkpts1, color)

    #Gets the figure object
    fig = plt.gcf()
    #Chooses black text if top-left corner is bright, else white text (to keep it visible)
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    #Writes the text (main caption) at the top-left corner of the first image
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    #Same logic, but places small_text at th bottom-left corner; useful for adding metadata (image size, runtime, etc.)
    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(small_text), transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    #Saves the figure to disk with no extra whitespace
    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    #Closes the figure to free memory (important if plotting many matches in a loop)
    plt.close()


#This is the OpenCV-optimized, fast version of the matching visualization; This is for real-time use (e.g., live camera matching)
def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    #Allocates a white canvas (255) big enough to hold both images
    out = 255*np.ones((H, W), np.uint8)
    #Places image0 on the left, image1 on the right, separated by margin
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    #Expands to RGB for drawing coloured matches
    out = np.stack([out]*3, -1)

    if show_keypoints:
        #Converts keypoints to integer pixel coords
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            #Draws black circles (radius 2) with white centers (radius 1) -> outlined style
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        #For right image keypoints, shifts x by W0 + margin
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    #Converts matched keypoints to integers
    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    #Converts color (likely float RGB in [0,1]) to OpenCV's BGR format [0-255]
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        #Draws a line between match pairs across the margin
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    #Uses two layers of text: black background (thick)+white foreground (thin) -> improves visibility
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    #Uses the same double-layer trick for small debug text, but at the bottom-left corner
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    #Saves to file if path is given
    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        #Displays interactively in an OpenCV window if opencv_display=True
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    #Returns the final visualization array (out) for further processing
    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)