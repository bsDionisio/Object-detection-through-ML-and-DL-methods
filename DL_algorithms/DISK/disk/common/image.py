import torch, math, warnings, imageio
import torch.nn.functional as F
import numpy as np

from torch_dimcheck import dimchecked
from torchtyping import TensorType

@dimchecked
#Input: 3D tensor with 'C'=number of channels, 'H'=height, 'W'=width
#Output: Returns a 3D tensor with the same number of channels but with a new height 'h' and width 'w'
def _rescale(tensor: TensorType['C', 'H', 'W'], size) -> TensorType['C', 'h', 'w']:
    return F.interpolate(
        #Adds a batch dimension, so shape goes from [C,H,W] -> [1,C,H,W]
        tensor.unsqueeze(0),
        #Performs resizing of the tensor using bilinear interpolation
        size=size,
        mode='bilinear',
        #Makes interpolation behave more like common image resize functions
        align_corners=False,
    #Removes the batch dimension, bringing shape back to [C, h, w]
    ).squeeze(0)

@dimchecked
#Inputs: tensor=[C,H,W]; size=desired output spatial size (new_H,new_W); 
# value=constant value used for padding (default 0.->black pixels if this is an image)
def _pad(tensor: TensorType['C', 'H', 'W'], size, value=0.):
    #Figures out how much padding is needed along width(W) and height(H) so that the tensor reaches the desire size
    xpad = size[1] - tensor.shape[2]
    ypad = size[0] - tensor.shape[1]

    # not that F.pad takes sizes starting from the last dimension
    #If the original tensor is [C,H,W], the result becomes [C, new_H, new_W]
    padded = F.pad(
        tensor,
        (0, xpad, 0, ypad),
        mode='constant',
        value=value
    )

    #Ensures the padded tensor's spatial dimensions are exactly (new_H, new_W)
    assert padded.shape[1:] == tuple(size)
    return padded

class Image:
    @dimchecked
    def __init__(
        self,
        #The instrinsic matrix of the camera (focal lengths, principal point, skew); shape [3,3]
        K     : TensorType[3, 3],
        #The rotation matrix for the camera's pose in world space; shape [3,3]
        R     : TensorType[3, 3],
        #The translation vector for the camera's pose in world space; shape [3]
        T     : TensorType[3],
        #The RGB image itself; 3 channels, height H, width W; shape [3, H, W]
        bitmap: TensorType[3, 'H', 'W'],
        #The depth map aligned with the image (distance of each pixel from the camera); single-channel tensor
        depth, #[1, 'H', 'W'],
        #Path to the original image file
        bitmap_path: str
    ):
        #Intrinsics
        self.K = K
        #Rotation
        self.R = R
        #Translation
        self.T = T

        #RGB image data
        self.bitmap = bitmap
        #Depth map
        self.depth  = depth

        # save bitmap path for potential debugging purposes
        self.bitmap_path = bitmap_path

    @property
    def K_inv(self):
        #self.K is the intrinsic camera matrix (3x3); This property returns its inverse
        return self.K.inverse()

    @property
    def hwc(self):
        #self.bitmap is stored in channel-first format: [3,H,W] (PyTorch default); permute (1,2,0) reorders aes -> [H,W,3];
        #This porperty gives you a "regular" image tensor layout
        return self.bitmap.permute(1, 2, 0)

    @property
    def shape(self):
        #Shortcut to access the spatial resolution of the image without the channel dimension
        return self.bitmap.shape[1:]

    #Function to resize the image (and depth map) while keeping the camera intrinsics consistent
    def scale(self, size):
        '''
        Rescale the image to at most size=(height, width). One dimension is
        guaranteed to be equally matched
        '''
        
        #How much we'd need to shrink each dimension
        x_factor = self.shape[0] / size[0]  #current height / target height
        y_factor = self.shape[1] / size[1]  #current width / target width

        #Choose the smaller resize ratio so the image fits inside (height, width) without exceeding either dimension.
        #This guarantees aspect ratio is preserved
        f = 1 / max(x_factor, y_factor)
        #Whichever side was the "limiting factor" gets matched exactly to the requested size; the other side is scaled proportionally
        if x_factor > y_factor:
            new_size = (size[0], int(f * self.shape[1]))
        else:
            new_size = (int(f * self.shape[0]), size[1])

        #The intrinsic matrix k depeds on image scale (focal lengths and principal point are in pixel units)
        K_scaler = torch.tensor([
            [f, 0, 0],
            [0, f, 0],
            [0, 0, 1]
        ], dtype=self.K.dtype, device=self.K.device)
        #Multiplying by K_scaler rescales those parameters consistently with the image; This ensures projections and 
        # 3D geometry still work correctly after resizing
        K = K_scaler @ self.K
 
        #Uses _rescale (the bilinear interpolation function); Rescales both RGB image (bitmap) and depth map (if available)
        bitmap = _rescale(self.bitmap, new_size)
        if self.depth is not None:
            depth = _rescale(self.depth, new_size)
        else:
            depth = None

        #Keeps extrinsics (R,T) the same (camera pose doesn't change); Updates intrinsincs (K) to match scaling; 
        # Returns a new Image object with consistent geometry
        return Image(K, self.R, self.T, bitmap, depth, self.bitmap_path)

    #Instead of rescaling, it enlarges the image canvas by padding
    def pad(self, size):
        #Calls _pad; Pads the RGB image ([3,H,W]) with zeros -> black pixels; Resulting shape: [3, new_H, new_W]
        bitmap = _pad(self.bitmap, size, value=0)
        #If depth exists ([1,H,W]), pad it to (1, new_H, new_W); missing areaas are filled with NaN, not 0
        if self.depth is not None:
            depth  = _pad(self.depth, size, value=float('NaN'))
        else:
            depth = None

        #Returns a new Image object with padded data; only the canvas gets larger
        return Image(self.K, self.R, self.T, bitmap, depth, self.bitmap_path)

    #This method is a device/dtype transfer helper for the Image class
    def to(self, *args, **kwargs):
        # use getattr/setattr to avoid repetitive code.
        # exclude `self.bitmap` because we don't need it on GPU (it's treated
        # separately by the dataloader)
        #The tensors that should follow the model/device: K=camera intrinsics; R=rotation; T=translation; depth=depth map
        TRANSFERRED_ATTRS = ['K', 'R', 'T', 'depth']

        #Loop over attributes
        for key in TRANSFERRED_ATTRS:
            attr = getattr(self, key)
            if attr is not None:
                attr_transferred = attr.to(*args, **kwargs)
            setattr(self, key, attr_transferred)

        return self

    @dimchecked
    #Input: xy=pixel coordinates, shape [2,N]. Each column=(u,v) pixel location; 'N'=Number of pixels
    #Output: xyz_w = 3D points in world coordinates, shape [3, N]
    def unproject(self, xy: TensorType[2, 'N']) -> TensorType[3, 'N']:
        #Gets depth values for each pixel (u, v); It's the distance fromt he camera to the surface along the viewing ray
        depth = self.fetch_depth(xy)

        #This gives homogeneous pixel coordinates
        xyw = torch.cat([
            xy.to(depth.dtype),
            torch.ones(1, xy.shape[1], dtype=depth.dtype, device=xy.device)
        ], dim=0)

        #Result: [3, N] points in camera frame
        xyz = (self.K_inv @ xyw) * depth
        #This results in 3D points in world coordinates
        xyz_w = self.R.T @ (xyz - self.T[:, None])

        #Shape [3, N]: world-space 3D positions of the original pixel coordinates
        return xyz_w

    @dimchecked
    #Input: xyw = world coordinates of 3D points, shape [3, N]
    #Output: projected pixel coordinates, shape [2, N]
    def project(self, xyw: TensorType[3, 'N']) -> TensorType[2, 'N']:
        #Applies extrinsics (world->camera); R=rotation matrix, maps world -> camera orientation; T=translation vector, 
        # shifts world coordinates into the camera frame
        extrinsic = self.R @ xyw + self.T[:, None]
        #K=intrinsic matrix (focal length, principal point, skew)
        intrinsic = self.K @ extrinsic
        #Divided by the third coordinate w' (perspective divide); Result = normalized 2D pixel coordinates [u,v]; Shape [2, N]
        return intrinsic[:2] / intrinsic[2]

    @dimchecked
    #Input: xy, hape [2,N]->x and y pixel coordinates or N points
    #Output: Boolean mask, shape [N]
    def in_range_mask(self, xy: TensorType[2, 'N']) -> TensorType['N']:
        #self.shape = (H,W); h = height, w = width
        h, w = self.shape
        #xy has sape [2,N]; x=[N] array of horizontal pixel position; y=[N] array of vertical pixel positions
        x, y = xy

        #Result: Boolean tensor [N]
        return (0 <= x) & (x < w) & (0 <= y) & (y < h)

    @dimchecked
    #Input: xy, shape [2, N] -> pixel coordinates (x, y) for N locations
    #Output: depth values for those locations, shape [N]
    def fetch_depth(self, xy: TensorType[2, 'N']) -> TensorType['N']:
        #If the Image object doesn't have a depth map, it fails immediately
        if self.depth is None:
            raise ValueError(f'Depth is not loaded')

        #Mask of which points are inside the image bounds
        in_range = self.in_range_mask(xy)
        #Ensures coordinates are not NaN or inf
        finite = torch.isfinite(xy).all(dim=0)
        #Points that are both inside and finite
        valid_depth = in_range & finite
        #Extract only valid coordinates; Convert to integer indices (since tensors are indexed with integers, not floats)
        x, y = xy[:, valid_depth].to(torch.int64)
        #Creates an output tensor of length N (one per input coordinates)
        depth = torch.full(
            (xy.shape[1], ),
            #Fills it with NaN by default (marking "no depth")
            fill_value=float('NaN'),
            device=xy.device,
            dtype=self.depth.dtype
        )
        #For valid pixels, fetch depth from the stored depth map ([1, H, W]); 
        # Depth at location (y, x) is assigned back into the output array
        depth[valid_depth] = self.depth[0, y, x]

        #Final result: [N] depth values; Invalid or out-of-bounds pixels ramin NaN
        return depth