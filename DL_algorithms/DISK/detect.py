import torch, h5py, imageio, os, argparse
from torchtyping import TensorType
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_dimcheck import dimchecked

from disk import DISK, Features

class Image:
    def __init__(self, bitmap: TensorType['C', 'H', 'W'], fname: str, orig_shape=None):
        self.bitmap     = bitmap
        self.fname      = fname
        #If no orig_shape is provided, it defaults to the shape of the image (H, W) from the tensor
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    #This method creates and returns a new Image object resized to a target shape
    def resize_to(self, shape):
        #Creates a new instance of Image with the resized bitmap, the same filename and the current shape
        return Image(
            #Scales the image to the desired shape (H, W); Padding (pad)-> ensures the resized image exactly matches the requeted shape
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    #Input: xys: a tensor of shape [2, N], representing N 2D points; First row= -coordinates; Second row= y-coordinates
    #Outpu: A tuple: scaled: same shape [2, N] - the coordinates rescaled into the original image's coordinate system; 
    # mask: shape [N]- a boolean mask marking which points fall inside the image bounds
    def to_image_coord(self, xys: TensorType[2, 'N']) -> tuple[TensorType[2, 'N'], TensorType['N']]:
        #Calculates the scaling factor f needed to map coordinates from the resized space back to the original space
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        #Divides the points by f -> now scaled represents the oordinates in original image units
        scaled = xys / f

        #Gets the original image height(h) and width(w)
        h, w = self.orig_shape
        #Splits the scaled [2, N] tensor into: x: all x-coordinates; y: all y-coordinates
        x, y = scaled

        #Builds a boolean mask ([N]) that is True only when a point is inside the image boundaries; 
        # Conditions: 0 <= x < w; 0 <= y < h
        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        #Returns: The scaled points [2, N]; A boolean mask [N] telling you which points are valid inside the original image
        return scaled, mask

    def _compute_interpolation_size(self, shape):
        #How much the height was scaled (orig_H/new_H)
        x_factor = self.orig_shape[0] / shape[0]
        #How much the width was scaled (orig_W/new_W)
        y_factor = self.orig_shape[1] / shape[1]

        #A uniform scaling factor that prevents distortion
        f = 1 / max(x_factor, y_factor)

        #Chooses which dimensions (height or width) controls the scaling; If the image was streched more in height 
        # -> lock height to shape[0] and adjust width proportionally
        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        #Otherwise, lock width to shape[1] and adjust height proportionally
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        #Returns f: the uniform scaling factor (used in to_image_coord to rescale points); new_size: the dimension that 
        # the image should be interpolated to, while preserving aspect ratio
        return f, new_size

    @dimchecked
    #Input: image=a tensor with shape [channels, height, width]; shape: target (H, W) shape you want to resize to
    #Output: A new tensor [C, h, w] where h,w match the computed interpolation size
    def _interpolate(self, image: TensorType['C', 'H', 'W'], shape) -> TensorType['C', 'h', 'w']:
        #Size is the corrected (H, W) that preserves aspect ratio while fitting into the given shape
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            #Since image is [C,H,M], we temporarily add a batch dimension -> [1, C, H, W]
            image.unsqueeze(0),
            #Resizes the image to the target size
            size=size,
            #Smooth interpolation method, common for images
            mode='bilinear',
            #Ensures correct behaviour when scaling
            align_corners=False,
        #Removes the fake batch dimensions we added earlier -> Result is back to [C, h, w]
        ).squeeze(0)
    
    @dimchecked
    #Input: image=tensor [C,H,W] (already interpolated, but might not exactly match the requested shape); shape=target (H,W) desired
    #Output: A new Tensor [C, h, w] exactly matching shape
    def _pad(self, image: TensorType['C', 'H', 'W'], shape) -> TensorType['C', 'h', 'w']:
        #Difference in height
        x_pad = shape[0] - image.shape[1]
        #Differene in width
        y_pad = shape[1] - image.shape[2]

        #If either value is negative, it means the image is larger than the target shape
        #Padding can only add pixels, not remove them, so it raises an error
        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        #Image is padded only on the bottom and right until it matches exactly the requested (H, W) shape
        return F.pad(image, (0, y_pad, 0, x_pad))


class SceneDataset:
    def __init__(self, image_path, crop_size=(None, None)):
        #Path to the directory where images are stored
        self.image_path = image_path
        #Tuple (height,width); Default is (None, None) -> meaning "don't drop" unless specified later
        self.crop_size  = crop_size
        #List of all filenames available in image_path
        self.names = [p for p in os.listdir(image_path) \
                      if p.endswith(args.image_extension)]

    def __len__(self):
        #Returns the number of images available
        return len(self.names)

    def __getitem__(self, ix):
        #Retrieves the i-th filename from the sataset
        name   = self.names[ix]
        #Build its full path
        path   = os.path.join(self.image_path, name)
        #Uses imageio.imread to read the file into a NumPy array; np.ascontiguousarray ensures the data is stored 
        # in memory contiguously -> safer when converting to a PyTorch tensor
        img    = np.ascontiguousarray(imageio.imread(path))
        #Converts to torch tensor
        tensor = torch.from_numpy(img).to(torch.float32)

        #If image is grayscale (H*W)
        if len(tensor.shape) == 2: # some images may be grayscale
            #add a channel dimension (H*W*1); then expand that single channel to 3 channels -> H*W*3; Ensures all images are RGB-like
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)

        #Rearranges and normalizes
        bitmap              = tensor.permute(2, 0, 1) / 255.
        #Wraps the tensor in your custom Image class
        extensionless_fname = os.path.splitext(name)[0]
        #Uses filename without extension as the identifier
        image = Image(bitmap, extensionless_fname)

        #If crop_size was specified in the dataset constructor
        if self.crop_size != (None, None):
            #Resize the image (with your aspect-ratio-preserving _interpolate + _pad combo)
            image = image.resize_to(self.crop_size)

        #Now dataset[ix] returns an Image object containing: The processed tensor (C*H*W normalized); the filename; the original shape info
        return image

    @staticmethod
    def collate_fn(images):
        #Takes each Image object int he batch; Extracts its .bitmap (a [C,H,W] tensor); 
        # Stacks along a new dimension 0 -> [B,C,H,W] where B = batch size
        bitmaps = torch.stack([im.bitmap for im in images], dim=0)
        
        #Returns bitmaps->fast GPU-friendly batch of image data; images->the list of Image objects (so you still have filenames, 
        # original shapes, etc)
        return bitmaps, images

#Input: dataset=an instance of SceneDataset; save_path=directory where outputs will be stored (.h5 files)
def extract(dataset, save_path):
    #Build dataloader
    dataloader = DataLoader(
        dataset,
        #Process one image at a time
        batch_size=1,
        #Speeds up GPU transfer
        pin_memory=True,
        #Uses your custom batching function to return (bitmaps, images)
        collate_fn=dataset.collate_fn,
        #Parallel data loading
        num_workers=4,
    )

    #Choose feature extractor; NMS (Non-Maximum Suppression): keep only strongest local keypoints
    if args.mode == 'nms':
        extract = partial(
            model.features,
            kind='nms',
            window_size=args.window,
            cutoff=0.,
            n=args.n
        )
    #RNG (Random Sampling): pick random keypoints
    else:
        extract = partial(model.features, kind='rng')

    #Creates the output folder
    os.makedirs(os.path.join(save_path), exist_ok=True)
    #Stores 2D keypoints
    keypoint_h5   = h5py.File(os.path.join(save_path, 'keypoints.h5'), 'w')
    #Stores feature descriptors
    descriptor_h5 = h5py.File(os.path.join(save_path, 'descriptors.h5'), 'w')
    #(Optional) Stores detection confidence scores
    if args.detection_scores:
        score_h5 = h5py.File(os.path.join(save_path, 'scores.h5'), 'w')

    #Wraps dataloader in a progress bar
    pbar = tqdm(dataloader)
    for bitmaps, images in pbar:
        #Moves image tensors to GPU (DEV)
        bitmaps = bitmaps.to(DEV, non_blocking=True)

        #Calls the model in inference mode
        with torch.no_grad():
            #If the image size isn't divided b 16, the U-Net backbone fails and gives a clear error message how to fix it
            try:
                batched_features = extract(bitmaps)
            except RuntimeError as e:
                if 'U-Net failed' in str(e):
                    msg = ('Please use input size which is multiple of 16 (or '
                           'adjust the --height and --width flags to let this '
                           'script rescale it automatically). This is because '
                           'we internally use a U-Net with 4 downsampling '
                           'steps, each by a factor of 2, therefore 2^4=16.')

                    raise RuntimeError(msg) from e
                else:
                    raise

        #Loops over each feature (keypoints, descriptors, scores) and corresponding image
        for features, image in zip(batched_features.flat, images):
            #Moves features to CPU for saving
            features = features.to(CPU)

            #Keypoints come in crop space (resized image coordinates)
            kps_crop_space = features.kp.T
            #Converts them back to original image coordinates with image.to_image_coord; mask filters valid points inside image bounds
            kps_img_space, mask = image.to_image_coord(kps_crop_space)

            #Applies mask -> keep only valid points
            keypoints = kps_img_space.numpy().T[mask]
            descriptors = features.desc.numpy()[mask]
            scores      = features.kp_logp.numpy()[mask]

            #Sorts keypoints by descending score (strongest first)
            order = np.argsort(scores)[::-1]

            keypoints   = keypoints[order]
            descriptors = descriptors[order]
            scores      = scores[order]

            #Ensures that descriptors have correct dimension (desc_dim)
            assert descriptors.shape[1] == args.desc_dim
            #Ensures that keypoints are 2D
            assert keypoints.shape[1] == 2

            #Optionally cast descriptors to half precision to save space
            if args.f16:
                descriptors = descriptors.astype(np.float16)

            #Stores results in HDF5 files under the image's filename
            keypoint_h5.create_dataset(image.fname, data=keypoints)
            descriptor_h5.create_dataset(image.fname, data=descriptors)

            if args.detection_scores:
                score_h5.create_dataset(image.fname, data=scores)

            #Progress bar shows how many keypoints were extracted
            pbar.set_postfix(n=keypoints.shape[0])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        "Script for detection and description (but not matching) of keypoints. "
        "It processes all images with extension given by `--image-extension` found "
        "in `image-path` directory. Images are resized to `--height` x `--width` "
        "for internal processing (padding them if necessary) and the output "
        "coordinates are then transformed back to original image size."),
    
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--height', default=None, type=int,
        help='rescaled height (px). If unspecified, image is not resized in height dimension'
    )
    parser.add_argument(
        '--width', default=None, type=int,
        help='rescaled width (px). If unspecified, image is not resized in width dimension'
    )
    parser.add_argument(
        '--image-extension', default='jpg', type=str,
        help='This script ill process all files which match `image-path/*.{--image-extension}`'
    )
    parser.add_argument(
        '--f16', action='store_true',
        help='Store descriptors in fp16 (half precision) format'
    )
    parser.add_argument('--window', type=int, default=5, help='NMS window size')
    parser.add_argument(
        '--n', type=int, default=None,
        help='Maximum number of features to extract. If unspecified, the number is not limited'
    )
    parser.add_argument(
        '--desc-dim', type=int, default=128,
        help='descriptor dimension. Needs to match the checkpoint value.'
    )
    parser.add_argument(
        '--mode', choices=['nms', 'rng'], default='nms',
        help=('Whether to extract features using the non-maxima suppresion mode or '
              'through training-time grid sampling technique')
    )
    
    default_model_path = os.path.split(os.path.abspath(__file__))[0] + '/depth-save.pth'
    parser.add_argument(
         '--model_path', type=str, default=default_model_path,
        help="Path to the model's .pth save file"
    )
    parser.add_argument('--detection-scores', action='store_true')
    
    parser.add_argument(
        'h5_path',
        help=("Directory where keypoints.h5 and descriptors.h5 will be stored. This"
              " will be created if it doesn't already exist.")
    )
    parser.add_argument(
        'image_path',
        help="Directory with images to be processed."
    )
    args = parser.parse_args()
    mode = 'GPU'
    CPU   = torch.device('cpu')
    if mode == 'CPU':
        DEV = CPU
    else:
        DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = SceneDataset(args.image_path, crop_size=(args.height, args.width))
    
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # compatibility with older model saves which used the 'extractor' name
    if 'extractor' in state_dict:
        weights = state_dict['extractor']
    elif 'disk' in state_dict:
        weights = state_dict['disk']
    else:
        raise KeyError('Incompatible weight file!')
    model = DISK(window=8, desc_dim=args.desc_dim)
    model.load_state_dict(weights)
    model = model.to(DEV)
    
    described_samples = extract(dataset, args.h5_path)