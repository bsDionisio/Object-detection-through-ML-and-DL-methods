import math
import torch
from torch import nn


#A 2D extension of the standard sinusoidal positional encoding from the original Transformer paper, but adapted for images instead of sequences
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.

            d_model: the channel dimension (must be divisible by 4, since encoding splits into sin/cos for x and y)
            max_shape: maximum expected feature map size (H,W)
            temp_bug_fix: whether to use the corrected implemented of the frequency scaling (div_term)
        """
        super().__init__()

        #Empty buffer where encodings will be stored -> shape [C,H,W]
        pe = torch.zeros((d_model, *max_shape))
        #Stores y-coordinates (row indices) [1,H,W]
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        #Stores x-coordinates (column indices)  [1,H,W]
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            #This creates frequencies for the sinusoidal functions, same as in Transformers; 
            # Here, half the channels encode x-position and half encode y-position
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        #Fill in the sinusoidal encoding; Channels are divided into 4 groups:
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)   #0::4 -> sin(x)
        pe[1::4, :, :] = torch.cos(x_position * div_term)   #1::4 -> cos(x)
        pe[2::4, :, :] = torch.sin(y_position * div_term)   #2::4 -> sin/y
        pe[3::4, :, :] = torch.cos(y_position * div_term)   #3::4 -> cos(y)

        #Adds a batch dimension; register_buffer tells PyTorch this tensor is part of the module, but not trainable
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    #Adds the positional encoding croppedto match the actual spatial size (H,W) of the input
    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]