import torch, typing
import numpy as np
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import NpArray, Features, MatchedPairs
from disk.geom import distance_matrix

class CycleMatcher:
    @dimchecked
    #Input: feat_1=Tensor of size [N,F] (features from image1);feat_2=tensor of size [M,F] (features from image2)
    #Ouput: a tensor of shape [2,K] representing K matches, where: Row0 contains indices from feat_1;
    #  Row1 contains corresponding indices from feat_2
    def match_features(self, feat_1: TensorType['N', 'F'], feat_2: TensorType['M', 'F']) -> TensorType[2, 'K']:
        #Produces an [N,M] matrix; entry (i,j)=distance between feat_1[i] and feat_2[j]
        dist_m = distance_matrix(feat_1, feat_2)

        #If either set has no features, fail early with an informative error message
        if dist_m.shape[0] == 0 or dist_m.shape[1] == 0:
            msg = '''
            Feature matching failed because one image has 0 detected features.
            This likely means that the algorithm has converged to a local
            optimum of detecting no features at all (0 reward). This can arise
            when lambda_fp and lambda_kp penalties are too high. Please check
            that your penalty annealing scheme is sound. It can also be that
            you are using a too low value of --warmup or --chunk-size
            '''
            raise RuntimeError(msg)
        #n_amin[i] gives the index j of the closest feature in feat_2 to feat_1[i]
        n_amin = torch.argmin(dist_m, dim=1)    #For each i in feat_1 -> best match in feat_2
        #m_amin[j] gives the index i of the closest feature in feat_1 to feat_2[j]
        m_amin = torch.argmin(dist_m, dim=0)    #For each j in feat_2 -> best match in feat_1

        # nearest neighbor's nearest neighbor
        nnnn = m_amin[n_amin]

        # we have a cycle consistent match for each `i` such that
        # nnnn[i] == i. We create an auxiliary array to check for that
        n_ix = torch.arange(dist_m.shape[0], device=dist_m.device)
        mask = nnnn == n_ix

        # Now `mask` is a binary mask and n_amin[mask] is an index array.
        # We use nonzero to turn `n_amin[mask]` into an index array and return
        return torch.stack([
            torch.nonzero(mask, as_tuple=False)[:, 0],  #indices in feat_1
            n_amin[mask],   #corresponding indices in feat_2
        ], dim=0)

    def match_pairwise(
        self,
        features: NpArray[Features], #a NumPy array of shape [N_scenes, N_per_scene]
    ): # -> [N_scenes, (N_per_scene choose 2)]
        #N_scenes=number of different scenes (datasets); N_per_scene=number of feature sets per scene
        #Calculates how many unique pairs of feature sets exist per scene
        N_scenes, N_per_scene = features.shape
        N_combinations        = N_per_scene * (N_per_scene - 1) // 2

        #Each element will hold a MatchedPairs object; Shape = [N_scenes, N_combinations]
        matched_pairs = np.zeros((N_scenes, N_combinations), dtype=object)
        
        for i_scene, scene_f in enumerate(features):
            i_decision = 0

            #Double loop ensures (i,j) with j>i, so each pair is considered exactly one; 
            # Extracts two Features objects: feature1 and feature2
            for i in range(N_per_scene):
                features1 = scene_f[i]
                for j in range(i+1, N_per_scene):
                    features2 = scene_f[j]

                    #Constructs a MatchedPairs object with:;    Saves the result in the matched_pairs array at the right index
                    matched_pairs[i_scene, i_decision] = MatchedPairs(
                        features1.kp,   #keypoints from the first feature set
                        features2.kp,   #keypoints from the second feature set
                        #Calls self.match_features to find correspondences between features1 and features2
                        self.match_features(features1.desc, features2.desc),    #matches returned by matched_pairs
                    )
                    
                    #Advance the pair index
                    i_decision += 1

        #Returns all pairwise matches for all scenes
        return matched_pairs

class CycleRatioMatcher:
    def __init__(self, ratio=0.8):
        self.ratio = ratio

    def __call__(self, dist_matrix):
        # Step 1: nearest neighbors for descriptors in image1
        nn12 = dist_matrix.topk(2, dim=1, largest=False)
        idx12, dists12 = nn12.indices, nn12.values

        # Step 2: apply Lowe's ratio test
        mask12 = dists12[:, 0] < self.ratio * dists12[:, 1]

        # Step 3: nearest neighbors for descriptors in image2
        nn21 = dist_matrix.argmin(dim=0)

        # Step 4: enforce cycle consistency
        idx1 = torch.arange(dist_matrix.size(0))
        idx2 = idx12[:, 0]
        cycle = (nn21[idx2] == idx1) & mask12

        # Return final matches
        matches = torch.stack([idx1[cycle], idx2[cycle]])
        return matches
