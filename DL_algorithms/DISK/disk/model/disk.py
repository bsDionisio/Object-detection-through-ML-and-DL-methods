import torch
import numpy as np

from torch_dimcheck import dimchecked
from torchtyping import TensorType
from unets import Unet, thin_setup

from disk import NpArray, Features
from disk.model.detector import Detector

DEFAULT_SETUP = {**thin_setup, 'bias': True, 'padding': True}

#Inherits from torch.nn.Module, making it compatible with the PyTorch training ecosystem
class DISK(torch.nn.Module):
    def __init__(
        self,
        desc_dim=128,   #Dimension of the feature descriptors for keypoints. Each detected keypoint will be represented by a vector of this size
        window=8,   #Used by Detector module for defining the local neighborhood when identifying keypoints
        setup=DEFAULT_SETUP,    #Likely a configuration dictionary/hyperparameter set for the Unet
        kernel_size=5,  #Used for the convolutional filters inside the Unet
    ):
        #Initializes the network
        super(DISK, self).__init__()

        self.desc_dim = desc_dim
        #A UNet-style convolutional neural network that takes 3 input channels (RGB image)
        self.unet = Unet(
            in_features=3, size=kernel_size,
            down=[16, 32, 64, 64, 64],  #Encoder progressively increases channels: 16->32->64->64->64
            up=[64, 64, 64, desc_dim+1],    #Decoder gradually reconstructs, ending with desc_dim+1 channels
            setup=setup,
        )
        #This module will take the heatmap and extract keypoint positions, 
        # using the given local window size to enforce non-maximum suppression or spatial filtering
        self.detector = Detector(window=window)

    @dimchecked
    #Input: unet_output of shape [B,C,H,W]
    def _split(self, unet_output: TensorType['B', 'C', 'H', 'W']) \
                -> tuple[TensorType['B', 'C-1', 'H', 'W'], TensorType['B', 1, 'H', 'W']]:
        '''
        Splits the raw Unet output into descriptors and detection heatmap.
        '''
        assert unet_output.shape[1] == self.desc_dim + 1

        #Splits the UNet output into two parts:
        #First desc_dim channels -> [B, desc_dim, H, W]
        descriptors = unet_output[:, :self.desc_dim]
        #Last channel -> [B, 1, H, W]
        heatmap     = unet_output[:, self.desc_dim:]

        return descriptors, heatmap

    @dimchecked
    #It's the front-facing function of your DISK network: it takes an image batch, runs it through the UNet + detector
    #Returns: keypoints with descriptors
    def features(
        self,
        images: TensorType['B', 'C', 'H', 'W'],
        kind='rng', #either "rng" or "nms" (two different strategies for choosing keypoints)
        **kwargs    #extra arguments passed to the detector functions
    ) -> NpArray[Features]: #a NumPy array of feature objects, one per image in the batch
        ''' allowed values for `kind`:
            * rng
            * nms
        '''

        #Stores number of images
        B = images.shape[0]
        try:
            descriptors, heatmaps = self._split(self.unet(images))
        except RuntimeError as e:
            if 'Trying to downsample' in str(e):
                msg = ('U-Net failed because the input is of wrong shape. With '
                       'a n-step U-Net (n == 4 by default), input images have '
                       'to have height and width as multiples of 2^n (16 by '
                       'default).')
                raise RuntimeError(msg) from e
            else:
                raise

        #It's a list (length B) of detected keypoints for each image
        keypoints = {
            'rng': self.detector.sample,
            'nms': self.detector.nms,
        }[kind](heatmaps, **kwargs)

        features = []
        for i in range(B):
            #Take the detected keypoints (keypoints[i])
            #Attach the corresponding local descriptors from descriptors[i]
            #Stores as a Features objects
            features.append(keypoints[i].merge_with_descriptors(descriptors[i]))

        #Returns a NumPy array of feature sts; Each entry corresponds to one image; 
        # Each feature set contains sparse keypoints with descriptors
        return np.array(features, dtype=object)