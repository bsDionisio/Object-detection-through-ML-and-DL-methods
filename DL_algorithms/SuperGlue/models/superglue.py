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
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn


def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron using 1D convolution layers"""
    #Number of layers(including input and output)
    n = len(channels)
    layers = []
    #For each layer, a 1D convolution is added
    for i in range(1, n):
        #kernel_size=1 means the convolution acts like a linear layer on the feature axis
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        #For all layers except the last, optionally apply batch normalization and ReLu activation
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    _, _, height, width = image_shape
    #Build tensor representing image size
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    #Compute image center
    center = size / 2
    #Compute scaling factor
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim: int, layers: List[int]) -> None:
        super().__init__()
        #Input dimension is 3 -> (x;y)=2coordinates, score=1 scalar
        self.encoder = MLP([3] + layers + [feature_dim])
        #Initializes the bias of the last layer in the MLP to zero. Stabilizes early training
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        #Transpose coordinates and adds scores as a 1D channel
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        #Concatenates channels ad applys the MLP
        return self.encoder(torch.cat(inputs, dim=1))


def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    dim = query.shape[1]
    #Compute attention scores
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    #Applies softmax accross the M dimensions - each query attends to all keys.
    prob = torch.nn.functional.softmax(scores, dim=-1)
    #Weighted sum of values using attention probabilities.
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        #Each head will process dim = d_model/num_heads features
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        #Used after attention. Combines all heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        #Three independent Conv1d layers for projecting query, key and value inputs into head dimensions
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        batch_dim = query.size(0)
        #Projects inputs to Multi-Head Format; Splits input into num_heads of size dim
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        #Returns attention output x of shape:[B, dim, num_heads, L]
        x, _ = attention(query, key, value)
        #Meerges into a single tensor [B, d_model, L]
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        #Computes attention from x to source
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        #Takes the concatenation of x and the attention output, and produces an updated x
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        #x queries source and gets an attention-weighted "message"
        message = self.attn(x, source, source)
        #Concatenates with original input ans refines with MLP
        return self.mlp(torch.cat([x, message], dim=1))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: List[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        #Loop over attention layers - this allows model to alternate between intra-set reasoning (self-attention) 
        # and inter-set communication (cross-attention)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            #Each attention layer returns an update (delta) for each descriptor
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)
            #Residual connection -> add delta to the original desc, preserving identity path
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    #Initialize dual variables u,v in log-space
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    #This corresponds to alternating row and column normalizations in log-space
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    #This is log_P, the logarithm of the doubly stochastic matrix after iters iterations
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    #sizes as tensors, used later in normalization
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    #Sink points for elements that do not match
    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    #(M+1)x(N+1) matrix where the last row and column are for dustbins
    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    #Log of total number of elements (for normalization)
    norm = - (ms + ns).log()
    #Row marginals (size M+1)
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    #Column marginals (size N+1)
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    #Ensures both match sums are normalized over (M+N)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    #Logarithm of doubly stochastic matrix approximating the optimal transport plan (including dustbins)
    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    #Rescales the log-probabilities by the original normalization factor
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    #Gets the size of the target dimension, creates a 1D tensor of ones with the same dtype and device as x 
    # and performs a cumulative sum along the 0-th dimension
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.2,
    }

    def __init__(self, config):
        super().__init__()
        #Combines user-provided config with default config
        self.config = {**self.default_config, **config}

        #Encodes each keypoint by combining its spatial location and detection confidence into a descriptor using an MLP
        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        #Uses alterning self-attention and cross-attention layers; allows descriptors from both images to interact 
        # and propagate context, helping identify correspondences
        self.gnn = AttentionalGNN(
            feature_dim=self.config['descriptor_dim'], layer_names=self.config['GNN_layers'])

        #1D convolution to prject updated descriptors into final form; used before computing similarity scores
        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        #Used when a keypoint doesn't match anything - like an "unmatched" slot
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        #Loads pretrained weights
        assert self.config['weights'] in ['indoor', 'outdoor']
        path = Path(__file__).parent
        path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
        self.load_state_dict(torch.load(str(path)))
        print('Loaded SuperGlue model (\"{}\" weights)'.format(
            self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores0': kpts0.new_zeros(shape0),
                'matching_scores1': kpts1.new_zeros(shape1),
            }

        # Keypoint normalization based on the image size, so they're scale-invariant and centered
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)

        # Keypoint MLP encoder - combines keypoint position [x;y] and detection score into a vector
        #Adds this to the original descriptor (from SuperPoint) to enrich it with spatial contextual info.
        desc0 = desc0 + self.kenc(kpts0, data['scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['scores1'])

        # Multi-layer Transformer network - uses AttentionGNN to pass info within and between images
        #Improves descriptor with global context using attention (both self-and-cross-attention layers)
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection - uses 1x1 convolution to map each enriched descriptor to its final matching space
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        #Ensures bidirectional consistency
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        #Accepts matches only if they are mutual and the softmax probability is above a confidence threshold
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        #Assigns -1 to unmatched keypoints
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

        return {
            'matches0': indices0, #Match for each keypoint in image0
            'matches1': indices1, #Match for each keypoint in image1
            'matching_scores0': mscores0, #Soft confidence
            'matching_scores1': mscores1,
        }