import torch
import numpy as np
import torch.nn.functional as F

from torch.distributions import Categorical, Bernoulli
from torch_dimcheck import dimchecked
from torchtyping import TensorType

from disk import Features, NpArray
from disk.model.nms import nms

@dimchecked
#Input: values=a tensor of shape [...,T]; indices=a tensor of the same leading shape [...], containing integers between 0 and t-1
def select_on_last(values: TensorType[..., 'T'], indices: TensorType[...]) -> TensorType[...]:
    '''
    WARNING: this may be reinventing the wheel, but I don't know how to do
    it otherwise with PyTorch.

    This function uses an array of linear indices `indices` between [0, T] to
    index into `values` which has equal shape as `indices` and then one extra
    dimension of size T.
    '''
    #For each set of indices, it picks out the corresponding value from the last dimension of values
    return torch.gather(
        values,
        -1,
        indices[..., None]
    ).squeeze(-1)

@dimchecked
#Goal: Implements a caregorical proposal -> Bernoulli acceptance sampling scheme
def point_distribution(
    logits: TensorType[..., 'T']    #a tensor of unnormalized log-probabilities over a categorical distribution, with last dimension size T
) -> tuple[TensorType[...], TensorType[...], TensorType[...]]:
    '''
    Implements the categorical proposal -> Bernoulli acceptance sampling
    scheme. Given a tensor of logits, performs samples on the last dimension,
    returning
        a) the proposals
        b) a binary mask indicating which ones were accepted
        c) the logp-probability of (proposal and acceptance decision)
    '''

    proposal_dist = Categorical(logits=logits)
    #Samples an index from the categorical distribution; integers in [0, T-1], shape [...]
    proposals     = proposal_dist.sample()
    #Gets the log-probability of that sample; log-prob of each proposal, sahpe [...]
    proposal_logp = proposal_dist.log_prob(proposals)

    #Looks up the logic corresponding to the sampled proposal (from the last dimension); 
    # This becomes the parameter for a Bernoulli distribution
    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    accept_dist    = Bernoulli(logits=accept_logits)
    #Sample a Bernoulli
    accept_samples = accept_dist.sample()
    #Record the log-probability of this decision
    accept_logp    = accept_dist.log_prob(accept_samples)
    #Convert to a boolean mask (True=accepted)
    accept_mask    = accept_samples == 1.

    #Since both the proposal and the acceptance setp are stochastic, the joint log-probability is the sum of their log-probs
    logp = proposal_logp + accept_logp

    #proposals=sampled category indices; accept_mask=boolean mask; logp=total log probability of (proposal+acceptance)
    return proposals, accept_mask, logp

class Keypoints:
    '''
    A simple, temporary struct used to store keypoint detections and their
    log-probabilities. After construction, merge_with_descriptors is used to
    select corresponding descriptors from unet output.
    '''

    @dimchecked
    #xys=keypoint detections (pixel coordinates of interest points); 
    # logp=log-probabilities (how confident the model is about each keypoint)
    def __init__(self, xys: TensorType['N', 2], logp: TensorType['N']):
        self.xys  = xys #shape [N,2] (integer pixel locations for each keypoint; each row is (x,y))
        self.logp = logp    #shape [N] (log-probabilities for each keypoint)

    @dimchecked
    #Input: descriptors= a dense feature map of shape [C,H,W], C=feature dimension (channels), 
    # H,W=height and width of the feature map
    def merge_with_descriptors(self, descriptors: TensorType['C', 'H', 'W']) -> Features:
        '''
        Select descriptors from a dense `descriptors` tensor, at locations
        given by `self.xys`
        '''
        #Splits xys (shape [N,2] into x and y coordinates); x.shape == y.shape == [N]
        x, y = self.xys.T

        #Uses advanced indexing to sample descriptors at the (x,y) locations; Each keypoint now has its own feature vectors
        desc = descriptors[:, y, x].T
        #L2-normalizes each feature vector
        desc = F.normalize(desc, dim=-1)

        #Wraps everything into a new Features object: Keypoint coordinates (xys), cast to float; 
        # Normalized descriptors (desc); Confidence scores (logp)
        return Features(self.xys.to(torch.float32), desc, self.logp)

#This class is building blocks for a keypoint detector that works on a CNN heatmap
class Detector:
    #Just stores the window size; Default 8x8 tiles
    def __init__(self, window=8):
        self.window = window

    @dimchecked
    #Input: heatmap: [B,C,H,W] (B=batch size; C=number of channels; H,W=height and width of the feature map)
    #Output: [B,C,h,w,T] (h=H//v, number of vertical tiles); w=W//v (number of horizontal tiles); 
    # T=v*v (flattened pixels inside each tile)
    def _tile(self, heatmap: TensorType['B', 'C', 'H', 'W']) -> TensorType['B', 'C', 'h', 'w', 'T']:
        '''
        Divides the heatmap `heatmap` into tiles of size (v, v) where
        v==self.window. The tiles are flattened, resulting in the last
        dimension of the output T == v * v.
        '''
        #Extracts dimensions; v=tile size
        v = self.window
        b, c, h, w = heatmap.shape

        #Makes sure H and W are divisible by v; So the heatmap can be evenly split into tiles
        assert heatmap.shape[2] % v == 0
        assert heatmap.shape[3] % v == 0

        #Splits the height into chunks of size (.unfold(2, v, v)); Splits the width into chunks of size v (.unfold(3, v, v)); 
        # Flattens the last two dimensions (v,v) into a single dimension T=v*v
        #Each element along T is a pixel inside the tile, stored as a flat vector
        return heatmap.unfold(2, v, v) \
                      .unfold(3, v, v) \
                      .reshape(b, c, h // v, w // v, v*v)

    @dimchecked
    #Input: heatmap=shape [B,1,H,W], B=batch size, ingle channel (1) since it's a confidence heatmap
    #Output: A NumPy array of length B, where each element is a keypoints object
    def sample(self, heatmap: TensorType['B', 1, 'H', 'W']) -> NpArray[Keypoints]:
        '''
            Implements the training-time grid-based sampling protocol
        '''
        #Gets window size, device and shape
        v = self.window
        dev = heatmap.device
        B, _, H, W = heatmap.shape

        #Ensures the heatmap can be evenly divided into tiles
        assert H % v == 0
        assert W % v == 0

        # tile the heatmap into [window x window] tiles and pass it to
        # the categorical distribution.
        heatmap_tiled = self._tile(heatmap).squeeze(1)
        proposals, accept_mask, logp = point_distribution(heatmap_tiled)

        # create a grid of xy coordinates and tile it as well
        cgrid = torch.stack(torch.meshgrid(
            torch.arange(H, device=dev),
            torch.arange(W, device=dev),
        )[::-1], dim=0).unsqueeze(0)
        #Tiles the coordinate grid -> shape [1, 2, H//v, W//v, v*v]; 
        # Each tile now has the coordinates of its v*v pixels
        cgrid_tiled = self._tile(cgrid)

        # extract xy coordinates from cgrid according to indices sampled
        # before
        xys = select_on_last(
            self._tile(cgrid).repeat(B, 1, 1, 1, 1),
            # unsqueeze and repeat on the (xy) dimension to grab
            # both components from the grid
            proposals.unsqueeze(1).repeat(1, 2, 1, 1)
        ).permute(0, 2, 3, 1) # -> bhw2
         
        keypoints = []
        #Iterates per batch
        for i in range(B):
            #Uses accept_mask to filter proposals -> only keep accepted keypoints
            mask = accept_mask[i]
            #Creates a Keypoints object
            keypoints.append(Keypoints(
                xys[i][mask],   #coordinates where accepted
                logp[i][mask],  #corresponding log-probabilities
            ))

        #Returns a NumPy array of length B; Each entry is a Keypoints instance (not stacked into a tensor, 
        # because each image may have a variable number of accepted keypoints)
        return np.array(keypoints, dtype=object)

    @dimchecked
    #This method performs non-maximum suppresion (NMS)-based keypoint detection at inference time
    def nms(
        self,
        heatmap: TensorType['B', 1, 'H', 'W'],  #Single-channel confidence map for each image
        n=None, #Optional number of keypoints to keep (per image)
        **kwargs    #passed into the nms() function
    ) -> NpArray[Keypoints]:    #A NumPy array of Keypoints objects (one per batch)
        '''
            Inference-time nms-based detection protocol
        '''
        #Removes the channel dimension -> [B, H, W]
        heatmap = heatmap.squeeze(1)
        #Runs NMS on the heatmap
        nmsed = nms(heatmap, **kwargs)

        keypoints = []
        #Process each image independently
        for b in range(heatmap.shape[0]):
            #Gives (y,x) coordinates of surviving pixels
            yx   = nmsed[b].nonzero(as_tuple=False)
            #extracts their confidence values
            logp = heatmap[b][nmsed[b]] #[N] tensor of scores
            #flips (y,x) -> (x,y)
            xy   = torch.flip(yx, (1, ))    #[N,2] tensor of coordinates

            if n is not None:
                n_ = min(n+1, logp.numel())
                # torch.kthvalue picks in ascending order and we want to pick in
                # descending order, so we pick n-th smallest among -logp to get
                # -threshold
                minus_threshold, _indices = torch.kthvalue(-logp, n_)
                mask = logp > -minus_threshold

                xy   = xy[mask]
                logp = logp[mask]

            #Stores results as a Keypoints object
            keypoints.append(Keypoints(xy, logp))

        #Returns one Keypoints instance per batch element; NumPy object array is used 
        # because each image may have a different number of detections
        return np.array(keypoints, dtype=object)