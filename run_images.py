import cv2 as cv2
import numpy as np

#from my_DL import DL
from ML_algorithms.sift import Sift
from ML_algorithms.orb import Orb
from DL_algorithms.my_DL import DL 

from ML_algorithms.akaze import Akaze
#from surf import Surf
from ML_algorithms.brief import Brief
#from latch import Latch

logo = cv2.imread('data/logo.png')
frame = cv2.imread('data/frame.png')

my_object = DL()  #WORKING
#my_object = Sift()   #ALL DONE
#my_object = Akaze()  #WORKING
#my_object = Brief()  WORKING

#my_object = Orb()
#my_object = Kaze()

my_object.find_key_points_logo(logo)
my_object.find_key_points_frame(frame)
my_object.find_matches(frame)  #only one available for akaze

cv2.waitKey(0)

