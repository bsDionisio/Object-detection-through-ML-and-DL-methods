import torch, os, argparse, h5py, warnings, imageio
from torchtyping import TensorType
import numpy as np
from tqdm import tqdm
from disk.model import CycleMatcher

from torch_dimcheck import dimchecked

from disk.geom import distance_matrix

MAX_FULL_MATRIX = 10000**2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'h5_path',
        help=('Path to the .h5 artifacts directory (containing descriptors.h5 '
              'and keypoints.h5)')
    )
    parser.add_argument(
        '--f16', action='store_true',
        help=('Compute distance matrices in half precision (offers a '
              'substantial speedup with Turing and later GPUs).')
    )
    parser.add_argument(
        '--u16', action='store_true',
        help=('Store matches with as uin16. This won\'t work if you have '
              'more than ~65k features in an image, but otherwise saves '
              'disk space.')
    )
    parser.add_argument(
        '--rt', type=float, default=None,
        help='Ratio test value. Leave unspecified to perform no ratio test'
    )
    parser.add_argument(
        '--save-threshold', type=float, default=-float('inf'),
        help=('Don\'t save matches between a pair of images if less than '
              '--save-threshold were found.')
    )
    parser.add_argument(
        '--max-full-matrix', type=int, default=10000**2,
        help=('this is the biggest match matrix that will attempt to be '
              'computed allocated in memory. Matrices bigger than that will '
              'be split into chunks of at most this size. Reduce if your '
              'script runs out of memory.')
    )

    args = parser.parse_args()
    args.rt = args.rt if args.rt is not None else 1.

    MAX_FULL_MATRIX = args.max_full_matrix

    #Checks if a GPU with CUDA is available; if not, uses CPU (torch.device("cpu"))
    DEV   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #Prints a message showing the input file path and the device chosen (cuda or cpu)
    print(f'Processing {args.h5_path} with DEV={DEV}')


class H5Store:
    #Constructor that runs when you create an H5Store object
    def __init__(self, path, dtype=torch.float32):
        #Opens a .h5 file named descriptors inside that folder, read-only
        self.ds_file = h5py.File(os.path.join(path, 'descriptors.h5'), 'r')
        #Opens a .h5 file named keypoints inside that folder, read-only
        self.kp_file = h5py.File(os.path.join(path, 'keypoints.h5'), 'r')
        self.dtype   = dtype
    
    def keys(self):
        #Returns a list of all dataset keys inside descriptor.h5; HDF5 files work like a dictionary, where keys point to datasets.
        return list(self.ds_file.keys())

    def __getitem__(self, ix):
        #Reads the dataset corresponding to ix (key string)
        descriptors = self.ds_file[ix][()]
        #Converts the stored numpy array into a PyTorch tensor
        desc = torch.from_numpy(descriptors)
        #If the tensor's dtype doesn't match what you requested (torch.float32 by default), it warns you and converts it
        if desc.dtype != self.dtype:
            warnings.warn(f'Type mismatch: converting {desc.dtype} to {self.dtype}')
            return desc.to(self.dtype)

        return desc

    #Retrieves the keypoints for a given key (like an image name) from keypoints.h5; Returns them as a NumPy array, not a PyTorch tensor
    def get_kp(self, ix):
        return self.kp_file[ix][()]

#It takes two keys and returns them in sorted order; Ensures a consistent ordering regardless of how they were passed
def pair_key(key_1, key_2):
    #If second key is bigger than the first -> return (key_1, key_2) as-is
    if key_2 > key_1:
        return key_1, key_2
    #If the first key is bigger -> return (key_2, key_1) -> swap to maintain ascending order
    elif key_1 > key_2:
        return key_2, key_1
    #If neither of the above is true, the keys must be equal; in this case, it raises an error
    else:
        raise ValueError(f'Equal keys {key_1}, {key_2}')

@dimchecked
#It maps a binary mask and another index vector into a stacked 2D index tensor
def _binary_to_index(binary_mask: TensorType['N'], ix2: TensorType['M']) -> TensorType[2, 'M']:
    #Stacks the two 1D tensors into a 2*M tensor, where Row0 = indices where binary_mask is True; Row1 0 the values of i*2
    return torch.stack([
        #Finds indices where the mask is True (or 1); [:, 0] flattens into a 1D tensor of positions
        torch.nonzero(binary_mask, as_tuple=False)[:, 0],
        #This is passed in directly (a vector of indices of length M)
        ix2
    ], dim=0)

@dimchecked
def _ratio_one_way(dist_m: TensorType['N', 'M'], rt) -> TensorType[2, 'K']:
    #Find the 2 smallest distances per row; For each feature in A, we know its best and second-best matches in B
    val, ix = torch.topk(dist_m, k=2, dim=1, largest=False)
    #For each row: best distance / second-best distance; if the best match is much better than the second-best, 
    # then it's a reliable match
    ratio = val[:, 0] / val[:, 1]
    #True where the match passed the ratio test
    passed_test = ratio < rt
    #Selects the best-match column index (the one at position 0) only for rows that passed the test
    ix2 = ix[passed_test, 0]

    #Returns a list of (row_from_A, col_from_B) pairs of valid matches
    return _binary_to_index(passed_test, ix2)

@dimchecked
def _match_chunkwise(ds1: TensorType['N', 'F'], ds2: TensorType['M', 'F'], rt) -> TensorType[2, 'K']:
    #Computes chunk size -> max number of descriptors from ds2 can be processed at once
    chunk_size = MAX_FULL_MATRIX // ds1.shape[0]
    #Will hold results from each chunk
    matches = []
    #Tracks where the current chunk begins
    start = 0

    while start < ds2.shape[0]:
        #Extracts a slice of ds2 descriptors; next chunk starts at strat and ends at start+chunk_size
        ds2_chunk = ds2[start:start+chunk_size]
        #Distance matrix between all N descriptors in ds1 and this chunk of ds2; shape: [N, chunk_size]
        dist_m = distance_matrix(ds1, ds2_chunk)
        #Returns matches between ds1 and this ds2 chunk
        one_way = _ratio_one_way(dist_m, rt)
        #Shifts the indices by start to put them back into global ds2 coordinates
        one_way[1] += start
        #Save the matches for this chunk
        matches.append(one_way)
        #Move start forward
        start += chunk_size

    #Combines matches from all chunks into one [2, K_total] tensor
    return torch.cat(matches, dim=1)
    
@dimchecked
#It implements mutual matching (aka symmetric matching)
def _match(ds1: TensorType['N', 'F'], ds2: TensorType['M', 'F'], rt) -> TensorType[2, 'K']:
    #Size of the full distance matrix; Total number of pairwise distances if we compared everything
    size = ds1.shape[0] * ds2.shape[0]

    #Matches from ds1->ds2 (each feature in A finds its best match in B)
    fwd = _match_chunkwise(ds1, ds2, rt)
    #Matches from ds2->ds1 (each feature in B finds its best match in A)
    bck = _match_chunkwise(ds2, ds1, rt)
    #Swaps rws->so it hasthe same orientation as fwd (row0 = index in ds1, row1 = index in ds2)
    bck = torch.flip(bck, (0, ))

    #Combines forward and backward matches into one [2, k1+k2] tensor
    merged = torch.cat([fwd, bck], dim=1)
    #Removes duplicate columns; returns_count also tells us how many times each match appeared
    unique, counts = torch.unique(merged, dim=1, return_counts=True)

    #Only keeps matches that appeared twice: once in forward, once in backward; this enforces mututal nearest-neighbour + ratio test
    return unique[:, counts == 2]

def match(desc_1, desc_2, rt=1., u16=False):
    #_match does the full mutual ratio test (forward+backward); Returns a PyTorch tensor of shape [2,K], where:
    # Row0 = indices in desc_1; Row1 = indices in desc_2; K = number of surviving mutual matches
    matched_pairs = _match(desc_1, desc_2, rt)
    #Moves data to CPU (important if _match ran on GPU); converts to NumPy array for interoperability
    matches = matched_pairs.cpu().numpy()

    #By default, indices are int64 in NumPy; If you pss u16=True, they'll be cast to uint16; Useful when saving matches to disk or 
    # if you know your dataset won't exceed 65k features
    if u16:
        matches = matches.astype(np.uint16)

    #Returns a numpy.ndarray of shape (2,K), type int64 or uint16
    return matches

#This is the top-level loop that drives pairwise descriptor matching across a dataset
def brute_match(descriptors, hdf):
    #Sorted list of all image IDs
    keys = sorted(list(descriptors.keys()))

    #Number of unique pairs
    n_total = (len(keys) * (len(keys) - 1)) // 2
    saved = 0
    #A tqdm progress bar to track progress
    pbar = tqdm(total=n_total)

    #Outer loop: Loop over each image key_1
    for i, key_1 in enumerate(keys):
        #Loads its descriptors and moves them to the compute device (GPU if available)
        desc_1 = descriptors[key_1].to(DEV)
        #Ensures there's a group in the HDF5 file named key_1 (this will hold its matches with others)
        group  = hdf.require_group(key_1)
        #Inner loop: Pair key_1 with all later images (keys[i+1:]) to avoid duplicate/reversed pairs
        for key_2 in keys[i+1:]:
            #If this pair (key_1, key_2) is already saved in HDF5, skip it
            if key_2 in group.keys():
                continue

            #Load descriptors for key_2 onto the compute device
            desc_2 = descriptors[key_2].to(DEV)
            
            #Matching Step
            try:
                #(2,K) numpy array of mutua l matches
                matches = match(desc_1, desc_2, rt=args.rt, u16=args.u16)
                #number of matches
                n = matches.shape[1]

                #if n passes the minimum threshold, save them into the HDF5 file under group[key_2]
                if n >= args.save_threshold:
                    group.create_dataset(key_2, data=matches)
                    #Counts how many pairs got saved
                    saved += 1
            #If something fails (e.g., due to GPU memory exhaustion), skips
            except RuntimeError:
                print('Error, skipping...')
                n = 0

            #Advances the progress bar by one pair
            pbar.update(1)
            #Show info: left= which image we're currently on (key_1); s= how many pairs saved so far; 
            # n= number of matches found for the current pair
            pbar.set_postfix(left=str(key_1), s=saved, n=n)

    #Closes the progress bar after all pairs are processed
    pbar.close()

class MatcherWrapper:
    #Small helper class
    class InnerWrapper:
        def __init__(self):
            #If no ratio threshold (rt is None), it uses a plain CycleMatcher
            if args.rt is None:
                self._cycle_matcher = CycleMatcher()
            #If a threshold exists, it uses CycleRatioMatcher with that ratio
            else:
                #self._cycle_matcher = CycleRatioMatcher(args.rt)
                return

        #Ensures shapes are correct
        @dimchecked
        #Runs a raw pairwise match between two descriptors sets
        def raw_mle_match_pair(self, ds1: TensorType['N', 'F'], ds2: TensorType['M', 'F']) -> TensorType[2, 'K']:
            #N*M matrix of distances
            dist = distance_matrix(ds1, ds2, normalized=True)
            #Returns [2,K] index pairs of matches
            return self._cycle_matcher(dist)

    def __init__(self):
        #The outer MatcherWrapper just instantiates an InnerWrapper inside itself; 
        # So a MatcherWrapper object has .matcher which is the actual usable matcher
        self.matcher = MatcherWrapper.InnerWrapper()

if __name__ == '__main__':
    #Selects dtype: helf precision (float16) if --f16 flag is given, else float32
    dtype = torch.float16 if args.f16 else torch.float32
    #Opens descriptors with H5Store, reading descriptors.h5 and keypoints.h5 from the dataset path
    described_samples = H5Store(args.h5_path, dtype=dtype)

    #Opens (or creates) an HDF5 file matches.h5 in append mode inside the dataset path
    with h5py.File(os.path.join(args.h5_path, 'matches.h5'), 'a') as hdf:
        #Calls brute_match -> computes all pairwise matches and saves them into matches.h5
        brute_match(described_samples, hdf)