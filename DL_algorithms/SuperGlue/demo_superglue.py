import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)
import matplotlib.cm as cm

torch.set_grad_enabled(False)


if __name__ == '__main__':
    #Initialization of prser object. Setting description of the program that 
    #will  be shown when someone runs the script with --help
    parser = argparse.ArgumentParser(
        description='SuperGlue algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #The program will be expecting two strings that correspond to the two file paths
    parser.add_argument(
        '--input', type=str, nargs=2,
        help='Path to two image files provided')
    
    #***OUTPUT DIRECTORY***
    #Path to save the processed frames. If not provided, outputs are not saved.
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Directory where to write output frames (If None, no output)')

    #***IMAGE FILTERING***
    #File patterns to match images in a directory (e.g. png, jpg)
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    
    #***INPUT CONTROL***
    #Skips frames/images (e.g: --skip 2 = process every second image)
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    #Limits the number of frames/images to process
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    
    #***IMAGE RESIZING***
    #Resize input images: two number = resize to exact dimensions; 
    #one number = max side is resized, aspect ratio preserved; 
    #-1 = no resizing
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    #***SUPERGLUE OPTIONS***
    #Selects SuperGlue model weights based on environment type
    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='outdoor',
        help='SuperGlue weights')
    
    #***SUPERPOINT KEYPOINT SETTINGS***
    #Limit number of keypoints (-1 means keep all)
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    #Only keep keypoints above this confidence level
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    #Radius for suppressing nearby weaker keypoints
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    
    #***MATCHING CONFIGURATION***
    #Iterations used in Sinkhorn algorithm (part of SuperGlue's matching)
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    #Controls how confident the matching should be
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')
    
    #***DISPLAY AND DEVICE SETTINGS***
    #If set, visualizes keypoints on output
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    #Prevents GPU display, useful for headless environments
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    #Ignores GPU and forces interference on CPU
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    
    #Resize only based on width
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    #Resize to the exact size given
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    #Resize image whilpe preserving aspect ratio so that the 
    #larger dimension equals this value
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    #setting up the device (CPU OR GPU) for inference and configuring parameters
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    #Read images or video frames and present them in a uniform interface for fetching them on-by-one
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    frame, ret = vs.next_frame()       #frame = actual image, ret = boolean flag indicating success or failure
    assert ret, 'Error when reading the first frame (try different --input?)'

    #Converts frame to a tensor that is ready-to-use for a neural network
    frame_tensor = frame2tensor(frame, device)
    #dictionary containing keypoint data
    last_data = matching.superpoint({'image': frame_tensor})
    #creation of new dictionary by adding the suffix '0'
    last_data = {k+'0': last_data[k] for k in keys}
    #adding the image tensor to the dictionary under 'image0'
    last_data['image0'] = frame_tensor
    #cache current frame and image ID
    last_frame = frame
    last_image_id = 0

    #if output directory has been provided, indicates where outputs will be written 
    #and if it doesn't exist, creates the directory
    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)
    
    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()

    while True:
        frame, ret = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        #marks a checkpoints and labels it 'data'
        timer.update('data')
        #stem0 = previous image, stem1 = current image
        stem0, stem1 = last_image_id, vs.i - 1

        #Converts frame to a tensor that is ready-to-use for a neural network
        frame_tensor = frame2tensor(frame, device)
        #merges last_data with 'image1' to compute matches between the previous image (image0) and the current image (image1)
        pred = matching({**last_data, 'image1': frame_tensor})
        #Retrieves the keypoints from the previous image
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        #Retrieves the detected keypoints from image1 as predicted by the model
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        #Retrieves the match indices for keypoints in image0
        #Each value in catches points to the index of hte corresonding keypoint in image1; -1 = no match found
        matches = pred['matches0'][0].cpu().numpy()
        #Confidence scores for each match
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        #Selects the matched keypoints from the previous image using the boolean mask
        mkpts0 = kpts0[valid]
        #First, filters matches to get only valid matches indices; 
        #then with those, selects the corresponding keypoints from current image
        mkpts1 = kpts1[matches[valid]]
        #Colors higher and lower confidence marks differently
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]

        #These thresholds control how selective the system is regarding keypoints and matches
        #Get of keypoint detection threshold from SuperPoint's configuration
        k_thresh = matching.superpoint.config['keypoint_threshold']
        #Get of match confidence threshold from SuperPoint's configuration
        m_thresh = matching.superglue.config['match_threshold']

        #Displays the keypoint detection threshold, the match confidence threshold 
        #and the IDs of the two images being matched
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]

        #Creates a visualization of keypoint macthes
        out = make_matching_plot_fast(
            last_frame, frame, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text)
        
        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            #Setting a character string representing the key that was pressed; if no key pressed -> -1
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                #Takes the prediction from the current frame and prepares it to become the next reference
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                #Updates the ID of the new anchor frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                    # Increase/decrease match threshold by 0.05 each keypress.
                    d = 0.05 * (-1 if key == 'd' else 1)
                    matching.superglue.config['match_threshold'] = min(max(
                        0.05, matching.superglue.config['match_threshold']+d), .95)
                    print('\nChanged the match threshold to {:.2f}'.format(
                        matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        #This block of code saves the match visualization image to disk, only if the user has specified an output directory
        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()