import math
import torch
import torch.nn as nn

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid


#Refines coarse matches (coarse matching) to fine-grained keypoint correspondences
class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self):
        #This call serves to initialize the base nn.Module
        super().__init__()

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, WW, C]    features of image0 at coarse match locations
            feat1 (torch.Tensor): [M, WW, C]    features of image 1
            data (dict) dictionary storing auxiliary info, like image sizes and previous matches
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],   expected fine coordinates + uncertainty
                'mkpts0_f' (torch.Tensor): [M, 2],  refined keypoints
                'mkpts1_f' (torch.Tensor): [M, 2]}
        """
        #M=number coarse matches; WW=number of locations in flattened window; C=feature dimension
        M, WW, C = feat_f0.shape
        #Width of the 2D window (assumes square, so WW= W*W)
        W = int(math.sqrt(WW))
        #Image resolution ratio between initial and fine-level images
        scale = data['hw0_i'][0] / data['hw0_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),    #empty tensor
                'mkpts0_f': data['mkpts0_c'],   #use coarse matches directly
                'mkpts1_f': data['mkpts1_c'],
            })
            return

        #Picks the feature vector at the center of the patch in image0; shape: [M, C]
        feat_f0_picked = feat_f0_picked = feat_f0[:, WW//2, :]
        #Computes dot product similarity between feat_f0_picked and all features in feat_f1
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        #Applies scaled softmax (temperature = 1/sqrt(C)) to get  probability heatmap of the match; 
        # Reshapes heatmap to [M,W,W] for 2D spatial reasoning
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap; Uses DSNT (Differentiable Spatial to Numerical Transform) to get continuous coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        #Coordinate grid for patch; Computes variance of the heatmap to measure uncenrtainty of the predicted point
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        #Uncertainty scalar per match
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
        
        # for fine-level supervision; Saves [x, y, uncertainty] for each fine-level match
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)

    #Decorator: ensures this function runs without tracking gradients (no backprop)
    @torch.no_grad()
    #Uses the attributes set in forward
    def get_fine_match(self, coords_normed, data):
        #W=patch width; WW= patch area W*W (flattened size); C=feature dimension; scale=ratio between input and fine-resolution image
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        # mkpts0_f and mkpts1_f
        mkpts0_f = data['mkpts0_c']
        #Adjustment for image1 resolution
        scale1 = scale * data['scale1'][data['b_ids']] if 'scale0' in data else scale
        #Refine keypoints for image 1
        mkpts1_f = data['mkpts1_c'] + (coords_normed * (W // 2) * scale1)[:len(data['mconf'])]

        #Finalizes results by storing fine-level keypoints for both images
        data.update({
            "mkpts0_f": mkpts0_f,
            "mkpts1_f": mkpts1_f
        })