import torch
import torch.nn.functional as F
from torch_dimcheck import dimchecked
from torchtyping import TensorType

@dimchecked
#It implements non-maximum suppression (NMS) over a 2D signal; The idea is to keep only local maxima and suppress everything else
#window_size = size of the local neighborhood (must be odd). Determines how "local" the maxima are
#cutoff: optional threshold to discard low values (suppress weak peaks)
#Returns a boolean tensor (mask) of the same shape (B,H,W), where True marks local maxima
def nms(signal: TensorType['B', 'H', 'W'], window_size=5, cutoff=0.) -> TensorType['B', 'H', 'W']:
    #Pooling kernels need to be odd-size so that there's a well-defined center pixel
    if window_size % 2 != 1:
        raise ValueError(f'window_size has to be odd, got {window_size}')

    #scans a sliding window (wndow_size * window_size) across the signal
    _, ixs = F.max_pool2d(
        signal,
        kernel_size=window_size,
        stride=1,   #Wth stride and padding, it introduces an ouput of the same shape as the input
        padding=window_size // 2,
        return_indices=True,    #makes it return the index of the max element inside each pooling window
    )

    # FIXME UPSTREAM: a workaround wrong shape of `ixs` until
    # https://github.com/pytorch/pytorch/issues/38986
    # is fixed
    #Due to a PyTorch bug, sometimes ixs comes with an extra batch dimension. This squeezes it out
    if len(ixs.shape) == 4:
        assert ixs.shape[0] == 1
        ixs = ixs.squeeze(0)

    #Creaates a tensor of absolute linear indices arranged as an (H,W) grid; This represents the "true" index of each pixel
    h, w = signal.shape[1:]
    coords = torch.arange(h * w, device=signal.device).reshape(1, h, w)
    #For each pixel, checks whether that pixel was selected as the max in its neighborhood
    nms = ixs == coords

    #If cutoff is provided, it only keeps maxima whose signal value is above the threshold; Otherwise, it returns all local maxima
    if cutoff is None:
        return nms
    else:
        return nms & (signal > cutoff)