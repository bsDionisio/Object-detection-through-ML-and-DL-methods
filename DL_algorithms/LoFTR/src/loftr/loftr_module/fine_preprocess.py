import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


#This class is a feature preprocessing module designed for fine-level feature extraction and refinement;
#Inherits from torch.nn.Module, meaning it's a learnable neural network component; It processes fine-level features (feat_0, feat_1) 
# with optional coarse-level features (feat_c0, feat_c1) to prepare input for the fine-level matching stage
class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        #Whether to concatenate coarse-level features with fine-level ones
        self.cat_c_feat = config['fine_concat_coarse_feat']
        #Size of the local fine window to crop features from
        self.W = self.config['fine_window_size']

        #Dimensionality of coarse features
        d_model_c = self.config['coarse']['d_model']
        #Dimensionality of fine features
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        #If coarse features are used
        if self.cat_c_feat:
            #Projects coarse features into the fine feature dimension
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            #Merges coarse and fine deatures after concatenation
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    #Initializes weights using Kaiming initialization for stage training
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    #Input: feat_0, feat_1 = fine-level features from two images (shape [B,C,H,W]); feat_c0, feat_c1 = coarse-level features (same idea); 
    # data = dictionary with match info (predicted matches, indices, shapes)
    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        #The stride between fine and coarse feature maps; Used to unfold fine features around the coarse match locations
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        #If there are no predicted matches, return empty tensors
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # 1. unfold(crop) all local windows; it extracts sliding local windows of size WxW from feature maps
        #After unfolding, each local patch is rearranged into shape: [batch, num_patches, window_size², channels]
        #So every coarse-level location corresponds to a flattened WxW patch in the fine feature map
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # 2. select only the predicted matches
        #Uses predicted correspondences (b_ids, i_ids, j_ids) to select only relevant fine-level patches
        #Shapes become [num_matches, W², d_model_f]
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids']]  # [n, ww, cf]
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            #Projects coarse features into fine dimension (down_proj)
            feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
                                                   feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            #Concatenates coarse + fine features (merge_feat) for richer context
            feat_cf_win = self.merge_feat(torch.cat([
                torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
                repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            ], -1))
            #Splits back into feat_f0_unfold and feat_f1_unfold
            feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)

        #Both are [num_matches, W², d_model_f], representing local fine-level windows centered at matched points, 
        # optionally fused with coarse context
        return feat_f0_unfold, feat_f1_unfold