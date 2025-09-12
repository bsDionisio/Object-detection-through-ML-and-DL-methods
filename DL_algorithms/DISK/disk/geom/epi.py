import torch

from torch_dimcheck import dimchecked
from torchtyping import TensorType

@dimchecked
#Creates a new tensor on the same device (CPU/GPU) and with the same dtype as v; 
# The return type TensorType [3,3] means a 3x3 matrix
def cross_product_matrix(v: TensorType[3]) -> TensorType[3, 3]:
    ''' following
        en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    '''

    return torch.tensor([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],     0]
    ], dtype=v.dtype, device=v.device)

@dimchecked
#This function converts 2D points into homogeneous 3D form (still on the image plane)
#Input: xy is a 2*N tensor of 2D points (pixel coordinates)
def xy_to_xyw(xy: TensorType[2, 'N']) -> TensorType[3, 'N']:
    #Adds a row of ones, making the points homogeneous coordinates
    ones = torch.ones(1, xy.shape[1], device=xy.device, dtype=xy.dtype)
    return torch.cat([xy, ones], dim=0)

@dimchecked
#This computes the essential matrix E between two camera views
def ims2E(im1, im2) -> TensorType[3, 3]:
    #Relative rotation from camera 1 to camera 2
    R = im2.R @ im1.R.T     #Camera rotation matrix (3 x 3)
    #Relative translation (camera 2 position in camera 1's frame)
    T = im2.T - R @ im1.T   #Camera translation vector (3 x 1)
    return cross_product_matrix(T) @ R

@dimchecked
#This function computes te fundamental matrix F; Fundamental matrix directly relates pixel coordinates in one image 
# to epipolar lines in the other, without needing knowledge of 3D structure
def ims2F(im1, im2) -> TensorType[3, 3]:
    E = ims2E(im1, im2)     #Essential matrix
    return im2.K_inv.T @ E @ im1.K_inv
 
@dimchecked
#This function computes a symetric epipolar distance between two sets of image points, which is a standard error in epipolar geometry
def symdimm(x1: TensorType[2, 'N'], x2: TensorType[2, 'M'], im1, im2) -> TensorType['N', 'M']:
    #Normalize points to camera coordinates; xy_to_xyw(x) converts 2D pixel coordinates to homogeneous coordinates;
    #Multiplying by K_inv converts pixel coordinates into normalized camera coordinates (removes intrinsic effects like focal length and principal point)
    x1n = im1.K_inv @ xy_to_xyw(x1)
    x2n = im2.K_inv @ xy_to_xyw(x2)

    #Encodes the relative pose (rotation + translation) between the two cameras; Is used to compute the epipolar constraint
    #If points are perfectly corresponding, this value is 0
    E = ims2E(im1, im2)

    E_x1  = E @ x1n     #epipolar lines in image 2 cprresponding to each x1
    Et_x2 = E.T @ x2n   #epipolar lines in image 1 corresponding to each x2
    x2_E_x1 = x2n.T @ E_x1  #measures algebric error of the epipolar constraint for all pairs of points; scalar products

    n = lambda v: torch.norm(v, p=2, dim=0)

    #Reciprocals of the norms, used to normalize the algebric error to approximate geometric (perpendicular) distance
    n1 = 1 / n(E_x1[:2])[None, :]   #n(E_x1[:2]) -> the Euclidean norm of the line vector [a,b]
    n2 = 1 / n(Et_x2[:2])[:, None]
    #Combines both directions (symmetric epipolar distance)
    norm = n1 + n2
    #x2_E_x1.pow(2) -> square of algebric error; Multiply by norm -> scales error to approximate the true perpendicular distance from points to epipolar lines
    dist = x2_E_x1.pow(2) * norm
    #Returns a matrix of shape (N,M) with symmetric epipolar distances between each pair of points
    return dist.T

@dimchecked
#This function computes an asymmetric epipolar distance, which is a simpler (non-symmetric) version of the epipolar error used in stereo vision
def asymmdist(x1: TensorType[2, 'N'], x2: TensorType[2, 'M'], F: TensorType[3, 3]) -> TensorType['N', 'M']:
    '''
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    '''

    #Converts 2D pixel coordinates (x,y) to homogeneous coordinates (x,y,1)
    x1_h = xy_to_xyw(x1)
    x2_h = xy_to_xyw(x2)

    #Computes the epipolar lines in image 1 corresponding to ponts x2 in image 2
    Ft_x2 = F.T @ x2_h      #F is the fundamental matrix relating the two images
    #Computes the Euclidean norm of the line vector [a,b] (ignoring c); norm is a 1 x M vector for each line
    norm  = torch.norm(Ft_x2[:2], p=2, dim=0)
    #Compute the asymmetric distance
    dist  = (Ft_x2 / norm).T @ x1_h
    #Returns a matrix of shape (N,M) where each entry measures the perpendicular distance from point x1[i] to the epipolar line of x2[j]
    return dist.T

@dimchecked
#Computes the asymmetric epipolar distance directly from two camera images
def asymmdist_from_imgs(x1: TensorType[2, 'N'], x2: TensorType[2, 'M'], im1, im2) -> TensorType['N', 'M']:
    #Computes the fundamental matrix between two cameras im1 and im2; Internally: Computes the essentl matrix from camera rotations and translation; Converts it 
    # to the fundamental matrix using the intrinsic matrices K of the cameras
    F = ims2F(im1, im2)
    #This computes the perpendicular distance from each point in x1 to the epipolar line corresponding to each point in x2; Returns an (N,M) matrix of distance
    return asymmdist(x1, x2, F)

@dimchecked
def p_asymmdist(x1: TensorType[2, 'N'], x2: TensorType[2, 'N'], F: TensorType[3, 3]) -> TensorType['N']:
    '''
    following http://www.cs.toronto.edu/~jepson/csc420/notes/epiPolarGeom.pdf
    (page 12)
    '''

    #Converts 2D points (x,y) into homogeneous coordinates (x,y,1)
    x1_h = xy_to_xyw(x1)    #Shape: (3,N)
    x2_h = xy_to_xyw(x2)    #Shape: (3,N)

    #Compute epipolar lines in image1
    Ft_x2 = F.T @ x2_h
    #Normalizes each line so that [a,b] has unit length; This makes the algebruc distance equal to perpendicular distance
    norm  = torch.norm(Ft_x2[:2], p=2, dim=0)
    Ft_x2_n = Ft_x2 / norm

    #Computes the sum of element-wise products along rows; This is the distance from point x1[i] to the epipolar line of the corresponding point x2[i]
    #Returns a 1D tensor of shape (N,)
    return torch.einsum('ca,ca->a', (Ft_x2_n, x1_h))

@dimchecked
def p_asymmdist_from_imgs(x1: TensorType[2, 'N'], x2: TensorType[2, 'N'], im1, im2) -> TensorType['N']:
    #Computes fundamental matrix F from two camera images (im1, im2)
    F = ims2F(im1, im2)
    #Calls p_asymmdist to get asymmetric distances for corresponding points
    return p_asymmdist(x1, x2, F)