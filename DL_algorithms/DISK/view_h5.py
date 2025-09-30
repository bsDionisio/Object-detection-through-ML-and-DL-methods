import argparse, h5py, os, imageio, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import time

from disk.common.vis import MultiFigure

parser = argparse.ArgumentParser(
    description='Script for viewing the keypoints.h5 and matches.h5 contents',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('h5_path', help='Path to .h5 artifacts')
parser.add_argument('image_path', help='Path to corresponding images')
parser.add_argument(
    '--image-extension', default='jpg', type=str,
    help='Extension of the images'
)
parser.add_argument(
    '--save', default=None, type=str,
    help=('If give a path, saves the visualizations rather than displaying '
          'them interactively')
)
parser.add_argument(
    'mode', choices=['keypoints', 'matches'],
    help=('Whether to dispay the keypoints (in a single image) or matches '
          '(across pairs)')
)

args = parser.parse_args()

#Keeps track of how many plots have been saved so far
save_i = 1
#This function uses global sabe_i do it can update the counter across calls
def show_or_save():
    #This function uses global save_i so it can update the counter across calls
    global save_i

    #If the program was run without a save path
    if args.save is None:
        #It just calls plt.show() to display the plot interactively and exits the function
        plt.show()
        return
    #If a save directory was provided
    else:
        #Builds a file path
        path = os.path.join(os.path.expanduser(args.save), f'{save_i}.png')
        #Saves the current plot
        plt.savefig(path)
        #Prints the path so you know where the file went
        print(f'Saved to {path}')
        #Increments save_i so the next plot will be saved
        save_i += 1
        #Closes the plot to free memory and avoid overlapping plots
        plt.close()

#Input: h5_path=the directory containing an HDF5 file; image_path=the directory containing corresponding image files
def view_keypoints(h5_path, image_path):
    #Opens keypoints.h5 in read-only mode; this file stores keypoints associated with images
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')

    fname_to_id = {}
    #Loops over all filenames inside the HDF5 file; tqdm shows a progress bar while iterating
    for filename in tqdm(list(keypoint_f.keys())):
        #Loads the actual NumPy array of keypoints for this image
        keypoints = keypoint_f[filename][()]

        #The filenames inside the HDF5 file don't have extensions, so the code adds one
        fname_with_ext = filename + '.' + args.image_extension
        #Joins it with image_path to get the full path
        path = os.path.join(image_path, fname_with_ext)
        #If the file doesn't exist, it raises an error
        if not os.path.isfile(path):
            raise IOError(f'Invalid image path {path}')

        #Loads the image with imageio
        image = imageio.imread(path)
        #Computes the scaling factor so the figure size is proportional to the image but normalized to a max dimensions
        scale = 10 / max(image.shape)
        #Creates a Matplotlib figure and axes of that size
        fig, ax = plt.subplots(figsize=(scale * image.shape[1], scale * image.shape[0]), constrained_layout=True)
        #Hides the axis ticks/labels (clean display)
        ax.axis('off')
        #Shows the image
        ax.imshow(image)
        #Plots the keypoints: Each point has coordinates (x,y) taken from the HDF5 array; While fill with a black outline for visibility
        ax.scatter(keypoints[:, 0], keypoints[:, 1], s=7, marker='o', color='white', edgecolors='black', linewidths=0.5)

        #Saves it as a numbered PNG in a specified directory
        show_or_save()

#Function that shows matches between pairs of images
def view_matches(h5_path, image_path):
    #Contains detected keypoints for each image
    keypoint_f = h5py.File(os.path.join(h5_path, 'keypoints.h5'), 'r')
    #Stores correspondences between pairs of images
    match_file = h5py.File(os.path.join(h5_path, 'matches.h5'), 'r')
    
    added = set()

    #For each image pair (key_1, key_2)
    for key_1 in match_file.keys():
        for key_2 in match_file[key_1].keys():
            #Load the array of matches (indices of keypoints that correspond)
            matches = match_file[key_1][key_2][()]
            #Load the actual keypoints from keypoints.h5
            kp_1 = keypoint_f[key_1][()]
            kp_2 = keypoint_f[key_2][()]

            #Builds full paths to both images
            path_1 = os.path.join(image_path, key_1 + '.' + args.image_extension)
            path_2 = os.path.join(image_path, key_2 + '.' + args.image_extension)

            #Loads them with imageio and wraps them as PyTorch tensors
            bm_1 = torch.from_numpy(imageio.imread(path_1))
            bm_2 = torch.from_numpy(imageio.imread(path_2))

            #Images may not have the same resolution
            bigger_x = max(bm_1.shape[0], bm_2.shape[0])
            bigger_y = max(bm_1.shape[1], bm_2.shape[1])

            #Pads both images (with black pixels by default) so they're the same size; Important so 
            # they can be displayed side by side in a consistent canvas
            padded_1 = F.pad(bm_1, (
                0, 0,
                0, bigger_y - bm_1.shape[1],
                0, bigger_x - bm_1.shape[0]
            ))
            padded_2 = F.pad(bm_2, (
                0, 0,
                0, bigger_y - bm_2.shape[1],
                0, bigger_x - bm_2.shape[0]
            ))

            #Sets up a figure showing the two images next to each other
            fig = MultiFigure(padded_1, padded_2)

            #Matches contains two arrays:   ; Looks up the actual (x,y) coordinates those matched points; Transposes (.T) them to 
            # the expected format for plotting
            # matches[0]= indices of keypoints in image 1
            left  = torch.from_numpy(kp_1[matches[0]]).T
            #matches[1] = indices of corresponding keypoints in image 2
            right = torch.from_numpy(kp_2[matches[1]]).T

            #Draws lines connecting corresponding keypoints across the two images
            fig.mark_xy(left, right)

                        # Statistics
            num_keypoints_1 = len(kp_1)
            num_keypoints_2 = len(kp_2)
            num_matches = len(matches[0])  # Number of matched keypoints

            # Compute the number of descriptors (assuming one descriptor per keypoint)
            num_descriptors_1 = num_keypoints_1 * kp_1.shape[1]  # e.g., 128 descriptors per keypoint if they are SIFT
            num_descriptors_2 = num_keypoints_2 * kp_2.shape[1]

            # Calculate memory usage (assuming each descriptor is a float32, i.e., 4 bytes)
            memory_1 = num_descriptors_1 * 4  # in bytes
            memory_2 = num_descriptors_2 * 4  # in bytes

            # Compute the matching score
            matching_score = num_matches / min(num_keypoints_1, num_keypoints_2)

            # Print the statistics
            print(f"Number of Keypoints Detected In The Reference Image: {num_keypoints_1}")
            print(f"Number of Keypoints Detected In The Current Image:   {num_keypoints_2}")
            print(f"Number of Matching Keypoints Between The Two Images: {num_matches}")
            print(f"Number of descriptors (reference): {num_descriptors_1}")
            print(f"Number of descriptors (current):   {num_descriptors_2}")
            print(f"Memory (reference descriptors):    {memory_1} bytes")
            print(f"Memory (current descriptors):      {memory_2} bytes")
            print("--- ROBUSTNESS METRIC ---")
            print(f"Matching Score: {matching_score:.4f} ({num_matches} matches / {min(num_keypoints_1, num_keypoints_2)} keypoints)")

            show_or_save()

if args.mode == 'keypoints':
    view_keypoints(args.h5_path, args.image_path)
elif args.mode == 'matches':
    view_matches(args.h5_path, args.image_path)