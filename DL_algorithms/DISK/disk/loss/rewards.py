import torch
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import Image
from disk.geom.epi import asymmdist_from_imgs

#The class is designed to evaluate how well pairs of keypoints from two images satisfy the epipolar constraint
class EpipolarReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th  #Distance thresold for epipolar constraint satisfaction
        self.lm_tp = lm_tp  #Reward given when the match is consistent with epipolar geometry
        self.lm_fp = lm_fp  #Penalty applied when the match is not consistent

    @dimchecked
    def __call__(
        self,
        kps1: TensorType['N', 2],   #Keypoints from image1, shape (N,2) (x,y pixel coords)
        kps2: TensorType['M', 2],   #Keypoints from image2, shape (M,2)
        img1: Image,    #The corresponding images (used for geometry, calibration, etc)
        img2: Image
    ) -> TensorType['N', 'M']:  #Returns a reward matrix of shape (N,M) assigning a score for each possible pair (kps1[i],kps2[j])
        '''
        assigns all pairs of keypoints across (kps1, kps2) a reward depending
        if the are correct or incorrect under epipolar constraints
        '''

        #Boolean mask of shape (N,M), where True means the pair (kps1[i],kps2[j]) passes the epipolar test
        good = self.classify(kps1, kps2, img1, img2)
        #self.lm_tp * good -> assigns the positive reward where matches are good;
        #self.lm_fp * (~good) -> assigns the negative penalty where matches are bad
        #The sum gives a full reward matrix
        return self.lm_tp * good + self.lm_fp * (~good)

    @dimchecked
    def classify(
        self,
        kps1: TensorType['N', 2],
        kps2: TensorType['M', 2],
        img1: Image,
        img2: Image,
    ) -> TensorType['N', 'M']:
        '''
        classifies all pairs of keypoints across (kps1, kps2) as correct or
        incorrect depending on epipolar error
        '''

        #Distances when projecting keypoints from img1 into img2's epipolar lines. Shape (N,M)
        epi_1_to_2 = asymmdist_from_imgs(kps1.T, kps2.T, img1, img2).abs()
        #Distances when projecting keypoints from img2 into img1's epipolar lines. Shape (M,N) before transpose
        epi_2_to_1 = asymmdist_from_imgs(kps2.T, kps1.T, img2, img1).abs()

        # the distance is asymmetric, so we check if both 2_to_1 is
        # correct and 1_to_2.
        return (epi_1_to_2 < self.th) & (epi_2_to_1 < self.th).T

#Evaluates whether keypoint matches between two images are geometrically correct by using depth-aware reprojection
class DepthReward:
    def __init__(self, th=2., lm_tp=1., lm_fp=-0.25):
        self.th   = th 
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

        #Creates an internal EpipolarReward instance for fallback classification when depth fails
        self._epipolar = EpipolarReward(th=th)

    @dimchecked
    def __call__(
        self,
        kps1: TensorType['N', 2],
        kps2: TensorType['M', 2],
        img1: Image,
        img2: Image
    ) -> TensorType['N', 'M']:
        '''
        classifies all (kp1, kp2) pairs as either
        * correct  : within dist_α in reprojection
        * incorrect: above dist_α away in epipolar constraints
        * unknown  : no depth is available and is not incorrect

        and assigns them rewards according to DepthReward parameters
        '''

        # reproject to the other image.
        kps1_r = img2.project(img1.unproject(kps1.T)) # [2, N]
        kps2_r = img1.project(img2.unproject(kps2.T)) # [2, M]

        # compute pixel-space differences between (kp1, repr(kp2))
        # and (repr(kp1), kp2)
        diff1 = kps2_r[:, None, :] - kps1.T[:, :, None] # [2, N, M]
        diff2 = kps1_r[:, :, None] - kps2.T[:, None, :] # [2, N, M]

        # NaNs indicate we had no depth available at this location
        has_depth = (torch.isfinite(diff1) & torch.isfinite(diff2)).all(dim=0)

        # threshold the distances
        close1    = torch.norm(diff1, p=2, dim=0) < self.th
        close2    = torch.norm(diff2, p=2, dim=0) < self.th
        
        #Marks pairs that are definitely bad under epipolar geometry
        epi_bad    = ~self._epipolar.classify(kps1, kps2, img1, img2)
        #Matches that are consistent under reprojection (depth-based)
        good_pairs = close1 & close2

        #Reward=lm_tp (positive) if good; Penalty=lm_fp (negative) if epipolar-inconsistent;
        #Pairs with no depth and not epipolar-inconsistent -> get 0 reward
        return self.lm_tp * good_pairs + self.lm_fp * epi_bad