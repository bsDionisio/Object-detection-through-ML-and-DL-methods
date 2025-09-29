import cv2 as cv2
import numpy as np

from ML_algorithms.sift import Sift
from ML_algorithms.orb import Orb
from ML_algorithms.akaze import Akaze
from ML_algorithms.brief import Brief
from my_DL import DL 

logo = cv2.imread('data/logo.png')
frame = cv2.imread('data/frame.png')

#my_object = DL() 
#my_object = Sift()
my_object = Akaze()
#my_object = Brief()
#my_object = Orb()

#my_object.find_key_points_logo(logo)
#my_object.find_key_points_frame(frame)
my_object.find_matches(frame)  #only one available for akaze

cv2.waitKey(0)

