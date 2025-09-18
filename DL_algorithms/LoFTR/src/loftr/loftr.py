import torch
import torch.nn as nn
from einops.einops import rearrange

from .backbone import build_backbone
from .utils.position_encoding import PositionEncodingSine
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .utils.coarse_matching import CoarseMatching
from .utils.fine_matching import FineMatching


#Inherits from nn,Module because it's a PyTorch model
class LoFTR(nn.Module):
    def __init__(self, config):
        super().__init__()  #Initializes the PyTorch model
        # Misc
        self.config = config    #Dictionary containing hyperparameters for the backbone, transformer layers, matching threshold, etc

        # Modules
        #A convolutional network that extracts multi-scale feature maps from the input images
        self.backbone = build_backbone(config)
        #Adds spatial information to feature for the transformer; Uses sine-based postional encoding like in the original Transformer paper
        self.pos_encoding = PositionEncodingSine(
            config['coarse']['d_model'],
            temp_bug_fix=config['coarse']['temp_bug_fix'])
        #A transformer that preocesses coarse-level features from both images; Captures global context and correlations from coarse matching
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        #Finds coarse correspondences between image features
        self.coarse_matching = CoarseMatching(config['match_coarse'])
        #Preparaes features for fine-level refinement, using the coarse matches
        self.fine_preprocess = FinePreprocess(config)
        #Refines the coarse atches to pixel-level accuracy
        self.loftr_fine = LocalFeatureTransformer(config["fine"])
        #Produces the final set of matched keypoints
        self.fine_matching = FineMatching()

    #Takes a dictionary data with images and optional masks
    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        #Extracts coarse (feats_c) and fine (feats_f) features
        #If images have the same size, concatenate for efficiency
        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_f = self.backbone(torch.cat([data['image0'], data['image1']], dim=0))
            (feat_c0, feat_c1), (feat_f0, feat_f1) = feats_c.split(data['bs']), feats_f.split(data['bs'])
        #If sizes differ, process images separately 
        else:  # handle different input shapes
            (feat_c0, feat_f0), (feat_c1, feat_f1) = self.backbone(data['image0']), self.backbone(data['image1'])

        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # 2. coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. match coarse-level features to estimate rough correspondences; Updates the data dictionary with coarse matches
        self.coarse_matching(feat_c0, feat_c1, data, mask_c0=mask_c0, mask_c1=mask_c1)

        # 4.Uses coarse matches to sample fine-level refinement patches around the matched locations;
        feat_f0_unfold, feat_f1_unfold = self.fine_preprocess(feat_f0, feat_f1, feat_c0, feat_c1, data)
        if feat_f0_unfold.size(0) != 0:  # at least one coarse level predicted
            feat_f0_unfold, feat_f1_unfold = self.loftr_fine(feat_f0_unfold, feat_f1_unfold)    #loftr_fine refines these to sub-pixel accuracy

        # 5. match fine-level
        self.fine_matching(feat_f0_unfold, feat_f1_unfold, data)

    #This method is overriding PyTorch's standard nn.Module.load_state_dict, which is used to load model weights from a checkpoint
    #Input: state_dict= Dictionary of parameter names -> tensors; The keys are layer names, and the values are the learned weights
    def load_state_dict(self, state_dict, *args, **kwargs):
        #It iterates over a copy of the keys
        for k in list(state_dict.keys()):
            #If a key starts with "matcher", remove that prefix
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        #Calls the original PyTorch loader with the modified dictionary; Handles strict/non-strict loading and other standard features
        return super().load_state_dict(state_dict, *args, **kwargs)