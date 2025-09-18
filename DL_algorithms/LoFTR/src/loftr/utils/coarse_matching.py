import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange

INF = 1e9

#This function fills a border of width b in all dimensions with a constant value
def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    #Do nothing if the border width is non-positive
    if b <= 0:
        return

    #Effectively, this sets all "edges" of the 4D structure to value v
    m[:, :b] = v    #Top along H0
    m[:, :, :b] = v #Left along W0
    m[:, :, :, :b] = v  #Top along H1
    m[:, :, :, :, :b] = v   #Left along W1
    m[:, -b:] = v   #Bottom along H0
    m[:, :, -b:] = v    #Right along W0
    m[:, :, :, -b:] = v #Bottom along H1
    m[:, :, :, :, -b:] = v  #Right along W1


#This one is similar, but it takes into account padding information
def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    #Mask the top/left borders first (like before)
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    #Compute valid region sizes from padding masks; p_m0 and p_m1 are padding masks for the first and second spatial grids
    #By summing and taking max, this extracts the effective sizes (h0,w0,h1,w1) of the valid regions per batch item
    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    #Instead of just masking the outer edge of the full tensor, this function masks the valid region's border, 
    # based on the actual size indicated by the padding mask
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v  #Bottom H0
        m[b_idx, :, w0 - bd:] = v   #Right W0
        m[b_idx, :, :, h1 - bd:] = v    #Bottom H1
        m[b_idx, :, :, :, w1 - bd:] = v #Right W1


#Computes the maximum number of valid candidates for all pairs of padded masks in a batch
def compute_max_candidates(p_m0, p_m1):
    """Compute the max candidates of all pairs within a batch
    
    Args:
        p_m0, p_m1 (torch.Tensor): padded masks; 3D tensors representing binary masks, where 1 means valid and 0 means padding
    """
    #p_m0.sum(1) -> sums over rows -> gives number of valid pixels per column for each batch item; 
    #p_m0.sum(-1) -> sum over columns -> gives number of valid pixels per row for each batch item;
    #max(-1)[0] > take the maximum across columns/rows -> gives the effective height (h0) and width (w0) for each mask in the batch
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]  #[N] tensor, height of valid region in p_m0 per batch item
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]  #[N] tensor, width of valid region in p_m0 per batch item
    #Gives the maximum number of pairs you can form per batch item, constrained by the smaller of the two masks;
    #Sum the per-batch maxima to get total maximum candidates for the batch
    max_cand = torch.sum(
        torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


#This class is designed to perform coarse-level matching between features of two images or feature maps
class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config    #Dictionary containing all hyperparameter
        # general config
        self.thr = config['thr']    #matching threshold - probably used to filter out low-confidence matches
        self.border_rm = config['border_rm']    #border removal width - probably used to remove matches near the image edges, where matches are often unreliable
        # -- # for training fine-level LoFTR
        self.train_coarse_percent = config['train_coarse_percent']  #percentage of top coarse matches to use during training
        self.train_pad_num_gt_min = config['train_pad_num_gt_min']  #minimum number of ground truth matches after padding (to avoid empty batches)

        # we provide 2 options for differentiable matching
        self.match_type = config['match_type']  #decides the differentiable matching strategy
        #Dual softmax is a differentiable matching method: Apply softmax along rows and then columns of a similarity matrix;
        #temperature controls the sharpness of the softmax distribution - lower means "more confident" matches
        if self.match_type == 'dual_softmax':
            self.temperature = config['dsmax_temperature']
        #Sinkhorn algorithm: computes a differentiable approximation of optimal transport for matching
        elif self.match_type == 'sinkhorn':
            try:
                from .superglue import log_optimal_transport
            except ImportError:
                raise ImportError("download superglue.py first!")
            self.log_optimal_transport = log_optimal_transport
            #Learnable parameter for Kinkhorn initialization
            self.bin_score = nn.Parameter(
                torch.tensor(config['skh_init_bin_score'], requires_grad=True))
            #Number of Sinkhorn iterations (more iterations -> more accurate but slower)
            self.skh_iters = config['skh_iters']
            #Optional pre-filtering of candidate matches to reduce computation
            self.skh_prefilter = config['skh_prefilter']
        #Ensures that only supported matching types (dual_softmax or sinkhorn) can be used
        else:
            raise NotImplementedError()

    #Core computation where coarse-level matches between teo sets of features are computed
    def forward(self, feat_c0, feat_c1, data, mask_c0=None, mask_c1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C] -> features from the first image (N=batch, L=number of keypoints, C=feature dim)
            feat1 (torch.Tensor): [N, S, C] -> features from the second image (S=number of keypoints)
            data (dict) -> stores outputs and intermediate results
            mask_c0 (torch.Tensor): [N, L] (optional) -> optional mask (1=valid, 0=padded) for first feature set
            mask_c1 (torch.Tensor): [N, S] (optional) -> optional mask for second feature set
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        #N=batch size; L=number of feature in feat_c0; S=number of features in feat_c1; C=feature dimension
        N, L, S, C = feat_c0.size(0), feat_c0.size(1), feat_c1.size(1), feat_c0.size(2)

        # normalize; scale features by sqrt(C) -> prevents the dot product from growing too large and stabilizes softmax later; 
        # Equivalent to scaled dot-product similarity used in Transformers
        feat_c0, feat_c1 = map(lambda feat: feat / feat.shape[-1]**.5,
                               [feat_c0, feat_c1])

        if self.match_type == 'dual_softmax':
            #Computes similarity matrix between features: [N,L,S]
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0,
                                      feat_c1) / self.temperature
            #Masks invalid positions using optional masks -> set similarity to -INF
            if mask_c0 is not None:
                sim_matrix.masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)
            #Dual software: softmax over rows and columns, then multiply -> produces a confidence score for each match; 
            #Ensures mutual consistency (row-wise and column-wise normalized)
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        elif self.match_type == 'sinkhorn':
            # sinkhorn, dustbin included; similar similarity matrix, with optional masking
            sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1)
            if mask_c0 is not None:
                sim_matrix[:, :L, :S].masked_fill_(
                    ~(mask_c0[..., None] * mask_c1[:, None]).bool(),
                    -INF)

            # build uniform prior & use sinkhorn
            #differentiable Sinkhorn algorithm; Adds dustbin row/column for unmatched points
            log_assign_matrix = self.log_optimal_transport(
                sim_matrix, self.bin_score, self.skh_iters)
            assign_matrix = log_assign_matrix.exp()
            #Removes dustbin, keeping only valid matches; Result: [N,L,S] confidence matrix
            conf_matrix = assign_matrix[:, :-1, :-1]

            # filter prediction with dustbin score (only in evaluation mode)
            #Evaluation mode only: filters out matches that are assigned to the dustbin (invalid)
            if not self.training and self.skh_prefilter:
                filter0 = (assign_matrix.max(dim=2)[1] == S)[:, :-1]  # [N, L]
                filter1 = (assign_matrix.max(dim=1)[1] == L)[:, :-1]  # [N, S]
                conf_matrix[filter0[..., None].repeat(1, 1, S)] = 0
                conf_matrix[filter1[:, None].repeat(1, L, 1)] = 0

            #Optional: keep full assignment including dustbin for supervision
            if self.config['sparse_spvs']:
                data.update({'conf_matrix_with_bin': assign_matrix.clone()})

        #Stores the coarse confidence matrix for downstream use
        data.update({'conf_matrix': conf_matrix})

        # predict coarse matches from conf_matrix; Extracts final matches, their coordinates, and confidence scores from conf_matrix
        data.update(**self.get_coarse_match(conf_matrix, data))

    #This means that it does not track gradients, because coarse match seletion is non-differentiable
    @torch.no_grad()
    #This method takes the coarse confidence matrix and extracts the actual coarse-level matches, 
    # optionally sampling them for training a fine-level matcher
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        #These are used for reshaping [L,S] vectors into [H0,W0,H1,W1] grids
        axes_lengths = {
            'h0c': data['hw0_c'][0],
            'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0],
            'w1c': data['hw1_c'][1]
        }
        _device = conf_matrix.device
        # 1. confidence thresholding; only keeps matches with confidence > self.thr
        mask = conf_matrix > self.thr
        #Reshape to 5D: [B, H0, W0, H1, W1] for spatial operations.
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c',
                         **axes_lengths)
        #Remove matches near the border to avoid unreliable matches; Supports optional padding masks
        if 'mask0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False,
                                     data['mask0'], data['mask1'])
        #Flatten back to [B, L, S] after border masking
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)',
                         **axes_lengths)

        # 2. mutual nearest; Keep only matches that are mutual best matches
        mask = mask \
            * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
            * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2) #finds if each row has at least one match
        b_ids, i_ids = torch.where(mask_v)  #indices of valid matches
        j_ids = all_j_ids[b_ids, i_ids] #column indices of matches
        mconf = conf_matrix[b_ids, i_ids, j_ids]    #confidence of selected matches

        # 4. Random sampling of training samples for fine-level LoFTR
        # (optional) pad samples with gt coarse-level matches
        if self.training:
            # NOTE:
            # The sampling is performed across all pairs in a batch without manually balancing
            # #samples for fine-level increases w.r.t. batch_size
            if 'mask0' not in data:
                num_candidates_max = mask.size(0) * max(
                    mask.size(1), mask.size(2))
            else:
                num_candidates_max = compute_max_candidates(
                    data['mask0'], data['mask1'])
            num_matches_train = int(num_candidates_max *
                                    self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert self.train_pad_num_gt_min < num_matches_train, "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=_device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (num_matches_train - self.train_pad_num_gt_min, ),
                    device=_device)

            # gt_pad_indices is to select from gt padding. e.g. max(3787-4800, 200)
            gt_pad_indices = torch.randint(
                    len(data['spv_b_ids']),
                    (max(num_matches_train - num_matches_pred,
                        self.train_pad_num_gt_min), ),
                    device=_device)
            mconf_gt = torch.zeros(len(data['spv_b_ids']), device=_device)  # set conf of gt paddings to all zero

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]],
                                       dim=0),
                *zip([b_ids, data['spv_b_ids']], [i_ids, data['spv_i_ids']],
                     [j_ids, data['spv_j_ids']], [mconf, mconf_gt]))

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        #Converts coarse indices to 2D coordinates in original image resolution; Accounts for scaling factors (scale0, scale1) if feature maps were downsampled
        scale = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,  #zero confidence indicates padded GT matches
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        #Returns: batch indices, keypoint indices, coordinates, and confidence
        return coarse_matches