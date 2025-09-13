import torch, math
import numpy as np
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import MatchedPairs, Image, NpArray
from disk.geom.epi import p_asymmdist_from_imgs

@dimchecked
def classify_pairs(kps1: TensorType['N', 2], kps2: TensorType['N', 2], img1: Image, img2: Image, th):
    '''
    classifies keypoint pairs as either possible or impossible under
    epipolar constraints
    '''

    #Computes the asymmetric epipolar distance
    epi_1_to_2 = p_asymmdist_from_imgs(kps1.T, kps2.T, img1, img2).abs()
    epi_2_to_1 = p_asymmdist_from_imgs(kps2.T, kps1.T, img2, img1).abs()

    #Only keep keypoint pairs that satisfy the epipolar constraint in both directions (symmetric check); 
    # The result is a boolean mask of length N: True->correspondence is geometrically plausible; False->correspondence violates epipolar geometry
    return (epi_1_to_2 < th) & (epi_2_to_1 < th)

#Purpose: Compute some discrete evaluation statistics
class DiscreteMetric(torch.nn.Module):
    #Input: th=Threshold for accepting/rejecting a match; lm_kp:Weight/penalty for keypoints; lm_tp:Weight/penalty for true positives; 
    # lm_fp:Weight/penalty for false positives
    def __init__(self, th=2., lm_kp=0., lm_tp=1., lm_fp=-0.25):
        super(DiscreteMetric, self).__init__() 

        self.th   = th
        self.lm_kp = lm_kp
        self.lm_tp = lm_tp
        self.lm_fp = lm_fp

    #Input: images=a 2D array of Image objects; matches=a 2D array of MatchedPairs
    def forward(
        self,
        images : NpArray[Image],       # [N_scenes, N_per_scene]
        matches: NpArray[MatchedPairs] # [N_scenes, N_per_scene choose 2]
    ):
        #Ensures matches are provided for every scene and every image pair
        N_scenes, N_per_scene = images.shape

        assert matches.shape[0] == N_scenes
        assert matches.shape[1] == ((N_per_scene - 1) * N_per_scene) // 2

        #Creates an array of smae shape as matches, filled with Python objects; 
        # Each entry will hold evaluation results for one pair of images
        stats = np.zeros(matches.shape, dtype=object)

        #Iteartes over all scenes, and within each sense over all unique pairs of images
        for i_scene in range(N_scenes):
            i_match = 0
            scene_matches = matches[i_scene]
            scene_images  = images[i_scene]

            #For each image pair:
            for i_image1 in range(N_per_scene):
                #Retrieves the corresponding match object
                image1 = scene_images[i_image1]

                for i_image2 in range(i_image1+1, N_per_scene):
                    image2 = scene_images[i_image2]

                    #Evaluates matches against epipolar/reprojection constraints; Counts TP/FP keypoints
                    stats[i_scene, i_match] = self._loss_one_pair(
                        scene_matches[i_match],
                        image1, image2
                    )

                    i_match += 1

        #stats contains evaluation results for each scene and image pair
        return stats

    #Input: pairs=A MatchedPairs object, holding pairs.kps1 (keypoints from img1), pairs.lps2 (keypoints from img2), 
    # pairs.matches(correspondences, indices into kps1 and kps2); img1, img2= The two images being compared
    def _loss_one_pair(self, pairs: MatchedPairs, img1: Image, img2: Image):
        #Simply adds up how many keypoints were detected across both images (not just matched ones)
        n_kps   = pairs.kps1.shape[0] + pairs.kps2.shape[0]

        #Pairs.matches is a (2, n_pairs) array of indices; pairs.matches[0] gives the indices into kps1; pairs.matches[1] gives the indices into kps2; 
        # This selects the subset of keypoints that are matched
        kps1 = pairs.kps1[pairs.matches[0]]
        kps2 = pairs.kps2[pairs.matches[1]]

        #Number of proposed matches between img1 and img2
        n_pairs = pairs.matches.shape[1]

        #Uses epipolar constraints (classify_pairs) to decide which correspondences are geometrically valid (good) or invalid (bad); 
        # good is a boolean mask of length n_pairs
        good = classify_pairs(kps1, kps2, img1, img2, th=self.th)
        bad  = ~good

        #How many matches passed the epipolar test (TPs)
        n_good = good.to(torch.int64).sum().item()
        #How many failed (FPs)
        n_bad  = bad.to(torch.int64).sum().item()
        #Precision=fraction of correct matches; The +1 prevents division by zero if no matches are presented
        prec   = n_good / (n_pairs + 1)

        #Weighted sum of: True positives (self.lm_tp * n_good)-> reward for good matches; False positives (self.lm_fp * n_bad)-> penalty for bad matches; 
        # Total keypoints (self.lm_kp * n_kps)-> baseline contribution, independent of matching
        reward = self.lm_tp * n_good  + \
                 self.lm_fp * n_bad   + \
                 self.lm_kp * n_kps

        #Packages everything into a dictionary
        stats = {
            'n_kps'    : n_kps,     #Number of keypoints considered
            'n_pairs'  : n_pairs,   #Number of proposed matches
            'tp'       : n_good,    #Number of geometriccally consistent matches
            'fp'       : n_bad,     #Number for inconsistent matches
            'reward'   : reward,    #Custom score
            'precision': prec,      #Correctness ratio
        }

        return stats