import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer, make_matching_plot_fast, frame2tensor)
import matplotlib.cm as cm
import time

torch.set_grad_enabled(False)


if __name__ == '__main__':
    #Initialization of prser object. Setting description of the program that 
    #will  be shown when someone runs the script with --help
    parser = argparse.ArgumentParser(
        description='SuperGlue algorithm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #The program will be expecting two strings that correspond to the two file paths
    path0 = "data/frame.png"
    path1 = "data/logo.png"

    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)
    
    #***OUTPUT DIRECTORY***
    #Path to save the processed frames. If not provided, outputs are not saved.
    output_dir = "results"

    #***IMAGE FILTERING***
    #File patterns to match images in a directory (e.g. png, jpg); Only interesting if dealing with video
    image_glob = ["frame.png", "logo.png"]

    #***INPUT CONTROL***
    #Skips frames/images (e.g: --skip 2 = process every second image)
    skip = 1    #no skipping needed
    #Limits the number of frames/images to process
    max_length = 100    #only process 100 frames
    
    #***IMAGE RESIZING***
    #Resize input images: two number = resize to exact dimensions; 
    resize = [-1]

    #***SUPERGLUE OPTIONS***
    #Selects SuperGlue model weights based on environment type
    superglue = 'outdoor'       #Choose between 'indoor' or 'outdoor'
    
    #***SUPERPOINT KEYPOINT SETTINGS***
    #Limit number of keypoints (-1 means keep all)
    max_keypoints = -1
    #Only keep keypoints above this confidence level
    keypoint_threshold = 0.005
    #Radius for suppressing nearby weaker keypoints
    nms_radius = 4
    
    #***MATCHING CONFIGURATION***
    #Number of iterations used in Sinkhorn algorithm (part of SuperGlue's matching)
    sinkhorn_iterations = 20
    #Controls how confident the matching should be
    match_threshold = 0.2
    
    #***DISPLAY AND DEVICE SETTINGS***
    #If set, visualizes keypoints on output
    show_keypoints = True   #True = visualize detected keypoints on the output
    #Prevents GPU display, useful for headless environments
    no_display =  False     #True = do not open GUI windows (useful on servers)
    #Ignores GPU and forces interference on CPU
    force_cpu = True   #True = ignore GPU and run everything on CPU


    #setting up the device (CPU OR GPU) for inference and configuring parameters
    device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    basedir = "data"

    #Read images or video frames and present them in a uniform interface for fetching them on-by-one
    vs = VideoStreamer(basedir, resize, skip, image_glob, max_length)
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
    if output_dir is not None:
        print('==> Will write outputs to {}'.format(output_dir))
        Path(output_dir).mkdir(exist_ok=True)
    
    # Create a window to display the demo.
    if not no_display:
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

        start_time = time.time()
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
            path=None, show_keypoints=show_keypoints, small_text=small_text)
        
        # Metric calculations
        num_kpts_logo = kpts0.shape[0]
        num_kpts_frame = kpts1.shape[0]
        num_matches = np.sum(valid)
        num_descriptors_logo = last_data['descriptors0'][0].nelement()
        num_descriptors_frame = pred['descriptors1'][0].nelement()
        memory_logo = num_descriptors_logo * 4  # float32 = 4 bytes
        memory_frame = num_descriptors_frame * 4
        
        # Matching score
        matching_score = num_matches / max(1, num_kpts_logo)  # Avoid division by zero
        
        # Print metrics
        print("\n--- METRICS ---")
        print("Number of Keypoints Detected In The Sample Image (logo): ", num_kpts_logo)
        print("Number of Keypoints Detected In The Overall Image (frame): ", num_kpts_frame)
        print("\nNumber of Matching Keypoints Between The Sample and Overall Images: ", num_matches)
        print("Overall Images: 1")
        print("Number of descriptors (logo):", num_descriptors_logo)
        print("Number of descriptors (frame):", num_descriptors_frame)
        print("Memory (logo descriptors): {:,} bytes".format(memory_logo))
        print("Memory (frame descriptors): {:,} bytes".format(memory_frame))
        print("--- ROBUSTNESS METRIC ---")
        print("Matching Score: {:.4f} ({} matches / {} keypoints)".format(matching_score, num_matches, num_kpts_logo))

        if not no_display:
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
                show_keypoints = not show_keypoints

        execution_time = time.time() - start_time
        timer.update('viz')
        timer.print()

        print("Execution time: {:.4f} seconds".format(execution_time))

        #This block of code saves the match visualization image to disk, only if the user has specified an output directory
        if output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)

    cv2.destroyAllWindows()
    vs.cleanup()