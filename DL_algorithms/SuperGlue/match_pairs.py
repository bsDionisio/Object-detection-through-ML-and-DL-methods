#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch


from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

torch.set_grad_enabled(False)


#This is the entry point of the code
if __name__ == '__main__':
    #Creates a command-line argument parser
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with SuperGlue',
        #Makes the help message show default values automatically
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_pairs', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
        help='Path to the list of image pairs')
    parser.add_argument(
        '--input_dir', type=str, default='assets/scannet_sample_images/',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='dump_match_pairs/',
        help='Path to the directory in which the .npz results and optionally,'
             'the visualization images are written')

    parser.add_argument(
        '--max_length', type=int, default=-1,
        help='Maximum number of pairs to evaluate')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--viz', action='store_true',
        help='Visualize the matches and dump the plots')
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform the evaluation'
             ' (requires ground truth pose and intrinsics)')
    parser.add_argument(
        '--fast_viz', action='store_true',
        help='Use faster image visualization with OpenCV instead of Matplotlib')
    parser.add_argument(
        '--cache', action='store_true',
        help='Skip the pair if output .npz files are already found')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Plot the keypoints in addition to the matches')
    parser.add_argument(
        '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
        help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
        '--opencv_display', action='store_true',
        help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--shuffle', action='store_true',
        help='Shuffle ordering of pairs before processing')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')

    #Reads arguments from the command line
    opt = parser.parse_args()
    print(opt)

    #Validation checks on the command-line arguments. These make sure the options passed don't conflict with each other. If it does, 
    # Python raises an AssertionError with the given message
    #If you want to display matches with OpenCV, you must also enable fast visualization; otherwise, the script wouldn't know what to display
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    #If you request OpenCVdisplay, you must also enble fast visualization; because OpenCV is only supported in the "fast" visualization mode
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    #If you want fast visualization, you also need to enable visualization; because fast visualization is just a variant of visualization
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    #If you use fast visualization, ou cannot request output in PDF format; because OpenCV only supports raster formats like .png
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    #If two numbers are passed to --resize but the second one is -1, it reduces it to a single number resize(resize max dimension only)
    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    #If two numbers were provided, it resizes every image to that exact width*height
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    #If the number given is positive, it resizes the image so that the largest side=640, preserving aspect ratio
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    #If --resize -1 was provided, it skips resizing completely
    elif len(opt.resize) == 1:
        print('Will not resize images')
    #If more than 2 numbers (invalid) were written, it raises an error
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    #Opens the pairs file; each line contains an image pair
    with open(opt.input_pairs, 'r') as f:
        #It splits each line into tokens (so pairs becomes a list of lists)
        pairs = [l.split() for l in f.readlines()]

    #If you set --max_length, keeps only that many pairs
    if opt.max_length > -1:
        pairs = pairs[0:np.min([len(pairs), opt.max_length])]

    #If you pass --shuffle, randomizes the order of pairs
    if opt.shuffle:
        random.Random(0).shuffle(pairs)

    #If you enable --eval mode, every line in input_pairs must have 38 tokens (filenames+ground-truth metadata)
    if opt.eval:
        #If not, it raises an error because evaluation requires pose+intrinsics info
        if not all([len(p) == 38 for p in pairs]):
            raise ValueError(
                'All pairs should have ground truth info for evaluation.'
                'File \"{}\" needs 38 valid entries per row'.format(opt.input_pairs))

    # Load the SuperPoint and SuperGlue models.
    #If you have a GPU and --force_cpu was not passed, it uses GPU; otherwise, it falls back to CPU
    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    #Prints out whic device is being used
    print('Running inference on device \"{}\"'.format(device))
    #This is a configuration object passed to the Matching model; contains two parts:
    config = {
        #Superpoint finds feature points in each image; nms_radius=non-maximum suppression radius (avois too-close keypoints);
        #keypoint_threshold=minimum confidence for keeping a keypoint; max_keypoints=max number ofkeypoints to detect (-1=keep all)
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        #SuperGlue takes SuperPoint's keypoints and matches them between images; weights=whichpretrained weights to use (indoor or outdoor);
        #sinkhorn_iterations=number of iterations for the Sinkhorn algorithm (used in optimal transport matching); match_threshold= how strict
        #the keypoints mathcing should be (higher=stricter)
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    #Creates a Matching object (a wrapper that runs SuperPoint+SuperGlue); .eval() puts the model in evalutaion mode (no training, disables dropout);
    #.to(device) moves the model to either CPU or GPU, dependingon the earlier check
    matching = Matching(config).eval().to(device)

    # Create the output directories if they do not exist already.
    #Converts the path (string from --input_dir) into a Path object (pathlib.Path), which makes file handling easier and OS-independent
    input_dir = Path(opt.input_dir)
    #Prints out where it will look for the input images
    print('Looking for data in directory \"{}\"'.format(input_dir))
    #Creates the output directory (--output_dir), if it doesn't lready exist
    output_dir = Path(opt.output_dir)
    #parents=True -> creates parent folders too if missing; exist_ok=True -> won't error if folder already exists
    output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write matches to directory \"{}\"'.format(output_dir))
    #If -eval is passed, it also tells you that evaluation results will be written into the same output_dir
    if opt.eval:
        print('Will write evaluation results',
              'to directory \"{}\"'.format(output_dir))
    #If --viz is passed, it tells you that visualization plots (images with keypoints and matches drawn) will also be saved in output_dir
    if opt.viz:
        print('Will write visualization images to',
              'directory \"{}\"'.format(output_dir))

    #Creates a timer utility to measure runtime per step
    timer = AverageTimer(newline=True)
    #Loops over all image pairs; Each pair is a list; the first two entries are filenames -> name0(first image), name1 (second image)
    for i, pair in enumerate(pairs):
        name0, name1 = pair[:2]
        #This selects the filename without the extensions
        stem0, stem1 = Path(name0).stem, Path(name1).stem
        #It creates four output file paths for each pair:
        #keypoints + matches results
        matches_path = output_dir / '{}_{}_matches.npz'.format(stem0, stem1)
        #evalutation results (pose errors, metrics)
        eval_path = output_dir / '{}_{}_evaluation.npz'.format(stem0, stem1)
        #visualization of matches
        viz_path = output_dir / '{}_{}_matches.{}'.format(stem0, stem1, opt.viz_extension)
        #visualization of evaluation results
        viz_eval_path = output_dir / \
            '{}_{}_evaluation.{}'.format(stem0, stem1, opt.viz_extension)

        # Handle --cache logic. Flags that decide whether to recompute or skip
        do_match = True
        do_eval = opt.eval
        do_viz = opt.viz
        do_viz_eval = opt.eval and opt.viz
        #If the user passed --cache, the script tries to reuse existing results instead of recomputing
        if opt.cache:
            #If *_matches.npz already exists -> load it with np.load
            if matches_path.exists():
                try:
                    results = np.load(matches_path)
                except:
                    raise IOError('Cannot load matches .npz file: %s' %
                                  matches_path)

                #Extract stored keypoints + matches + confidence
                kpts0, kpts1 = results['keypoints0'], results['keypoints1']
                matches, conf = results['matches'], results['match_confidence']
                do_match = False
            #If in evaluation mode and eval results already exist
            if opt.eval and eval_path.exists():
                try:
                    results = np.load(eval_path)
                except:
                    raise IOError('Cannot load eval .npz file: %s' % eval_path)
                #Load rotation error (err_R), tranlation error (err_t), precision, matching score, number of correct matches, epipolar errors
                err_R, err_t = results['error_R'], results['error_t']
                precision = results['precision']
                matching_score = results['matching_score']
                num_correct = results['num_correct']
                epi_errs = results['epipolar_errors']
                #Set do_eval=False -> skip recomputing evaluation
                do_eval = False
            #If visualization images already exist: skip creating them again
            if opt.viz and viz_path.exists():
                do_viz = False
            if opt.viz and opt.eval and viz_eval_path.exists():
                do_viz_eval = False
            #Records the time spent in this caching step
            timer.update('load_cache')

        #If all are False, then prints progress message and skips to the next pair
        if not (do_match or do_eval or do_viz or do_viz_eval):
            timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))
            continue

        # If a rotation integer is provided (e.g. from EXIF data), use it:
        if len(pair) >= 5:
            rot0, rot1 = int(pair[2]), int(pair[3])
        else:
            rot0, rot1 = 0, 0

        # Load the image pair.
        image0, inp0, scales0 = read_image(
            input_dir / name0, device, opt.resize, rot0, opt.resize_float)
        image1, inp1, scales1 = read_image(
            input_dir / name1, device, opt.resize, rot1, opt.resize_float)
        #If there was no image or it couldn't be read, prints an error and exits the script
        if image0 is None or image1 is None:
            print('Problem reading image pair: {} {}'.format(
                input_dir/name0, input_dir/name1))
            exit(1)
            #Updates the timer, recording how long it took to load this pair of images
        timer.update('load_image')

        #If matches for this pair don't already exist, we run the model
        if do_match:
            # Perform the matching. Input:a dictionary with two tensors (inp0 and inp1), already preprocessed; output(pred):a dict of 
            # predictions (keypoints, matches, confidence scores, etc.,), stored as PyTorch tensors
            pred = matching({'image0': inp0, 'image1': inp1})
            #For each entry in pred: v[0] because batch sie is 1; .cpu()->move from GPU to CPU memory; .numpy()->convert from 
            # PyTorch tensor to NumPy array;  Now, everything is NumPy arrays, easier for saving and post-processing
            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
            #Detected keypoints in image0 and image1
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            #Matches: For each keypoint in image0, index of its match in image1 (or -1 if no match); conf: confidence scores for each match
            matches, conf = pred['matches0'], pred['matching_scores0']
            #Logs how long the model inference took
            timer.update('matcher')

            # Write the matches to disk.
            out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                           'matches': matches, 'match_confidence': conf}
            #Saves to _matches.npz (compressed NumPy format) at the previously defined matches_path; This is used for catching 
            # and later visualization/evaluation
            np.savez(str(matches_path), **out_matches)

        # Keep the matching keypoints pairs that will be used for visualization and evaluation
        #Boolean mask. True only where a valid match exists
        valid = matches > -1
        #The subset of keypoints in image0 that were successfully matched
        mkpts0 = kpts0[valid]
        #The correspoding keypoints in image1 (using matches[valid] as indices)
        mkpts1 = kpts1[matches[valid]]
        #Confidence scores for these valid matches
        mconf = conf[valid]

        if do_eval:
            # Estimate the pose and compute the pose error. For evaluation, each pair must contain 38 tokens in the input file
            #These include: 2 filenames, 2 rotations, 9 intrinsic for image0 (3*3 matrix), 9 intrinsics for image1 (3*3 matrix),
            #16 extrinsics (4*4 relative transformation matrix from cam0 -> cam1)
            assert len(pair) == 38, 'Pair does not have ground truth info'
            #3*3 camera instrinsics matrices (fx, fy, cx, cy, etc.)
            K0 = np.array(pair[4:13]).astype(float).reshape(3, 3)
            K1 = np.array(pair[13:22]).astype(float).reshape(3, 3)
            #4*4 homogeneous transformation matrix from camera0 to camera1 (ground truth)
            T_0to1 = np.array(pair[22:]).astype(float).reshape(4, 4)

            # Scale the intrinsics to resized image, if images were resized
            K0 = scale_intrinsics(K0, scales0)
            K1 = scale_intrinsics(K1, scales1)

            # Update the intrinsics + extrinsics if EXIF rotation was found.
            if rot0 != 0 or rot1 != 0:
                cam0_T_w = np.eye(4)
                cam1_T_w = T_0to1
                if rot0 != 0:
                    K0 = rotate_intrinsics(K0, image0.shape, rot0)
                    cam0_T_w = rotate_pose_inplane(cam0_T_w, rot0)
                if rot1 != 0:
                    K1 = rotate_intrinsics(K1, image1.shape, rot1)
                    cam1_T_w = rotate_pose_inplane(cam1_T_w, rot1)
                cam1_T_cam0 = cam1_T_w @ np.linalg.inv(cam0_T_w)
                T_0to1 = cam1_T_cam0

            #Epipolar geometry errors for each match
            epi_errs = compute_epipolar_error(mkpts0, mkpts1, T_0to1, K0, K1)
            #A match is considered correct if error < 5e-4
            correct = epi_errs < 5e-4
            #How many matches passed
            num_correct = np.sum(correct)
            #Fraction of valid matches that are correct
            precision = np.mean(correct) if len(correct) > 0 else 0
            #num_correct / total keypoints in image0
            matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0

            thresh = 1.  # In pixels relative to resized image size.
            #Uses RANSAC-based pose estimation with an inlier threshold of 1 pixel
            ret = estimate_pose(mkpts0, mkpts1, K0, K1, thresh)
            #If pose can't be estimated -> set errors to infinity
            if ret is None:
                err_t, err_R = np.inf, np.inf
            #If successful, returns estimated rotation R, translation t, and inliers; computes rotation error (err_R) and
            #translation error (err_t) against ground truth T_0to1
            else:
                R, t, inliers = ret
                err_t, err_R = compute_pose_error(T_0to1, R, t)

            # Write the evaluation results to disk.
            out_eval = {'error_t': err_t,
                        'error_R': err_R,
                        'precision': precision,
                        'matching_score': matching_score,
                        'num_correct': num_correct,
                        'epipolar_errors': epi_errs}
            np.savez(str(eval_path), **out_eval)
            #Records how long evalutation took for profiling
            timer.update('eval')

        #Controlled by the command-line flag --viz
        if do_viz:
            # Visualize the matches.
            #mconf=confidence scores of the matches (output of SuperGlue); cm.jet(): a calourmap from matplotlib that maps match
            #confidence to colours (blue=low confidence, red=high confidence)
            color = cm.jet(mconf)
            #Text displayed on the visualization
            text = [
                'SuperGlue',
                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                'Matches: {}'.format(len(mkpts0)),
            ]
            #Optional roation info (if EXIF rotation was applied)
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info.
            #Keypoint detection threshold used by SuperPoint
            k_thresh = matching.superpoint.config['keypoint_threshold']
            #Match threshold used by SuperGlue
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            #Call the visualization function; Draws side-by-side images with lines connecting matched keypoints, and saves/shows them
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0, mkpts1, color,
                text, viz_path, opt.show_keypoints,
                opt.fast_viz, opt.opencv_display, 'Matches', small_text)

            #Records how long it took to generate the visualization
            timer.update('viz_match')

        if do_viz_eval:
            # Visualize the evaluation results for the image pair.
            color = np.clip((epi_errs - 0) / (1e-3 - 0), 0, 1)
            color = error_colormap(1 - color)
            #Labels depend on visualization style: -fast_viz=True -> palin text (deg, Delta) for OpenCV quick plots
            #-fast_viz=False -> LaTeX-like symbols for high-quality Matplotlib plots
            deg, delta = ' deg', 'Delta '
            if not opt.fast_viz:
                deg, delta = '°', '$\\Delta$'
            #Displays translaton error (err_t) and rotation error (err_R); if pose estimation failed -> shows "FAIL"; 
            # otherwise formats with 1 decimal place and the correct unit (degrees)
            e_t = 'FAIL' if np.isinf(err_t) else '{:.1f}{}'.format(err_t, deg)
            e_R = 'FAIL' if np.isinf(err_R) else '{:.1f}{}'.format(err_R, deg)
            #Overlay text info on visualization: SuperGlue (method used), rotation error, translation error, inliers count (correct matches / total matches)
            text = [
                'SuperGlue',
                '{}R: {}'.format(delta, e_R), '{}t: {}'.format(delta, e_t),
                'inliers: {}/{}'.format(num_correct, (matches > -1).sum()),
            ]
            #Adds EXIF rotation info if images had been rotated before matching
            if rot0 != 0 or rot1 != 0:
                text.append('Rotation: {}:{}'.format(rot0, rot1))

            # Display extra parameter info (only works with --fast_viz).
            #Keypoint confidence threshold (SuperPoint)
            k_thresh = matching.superpoint.config['keypoint_threshold']
            #Match confidence threshold (SuperGlue)
            m_thresh = matching.superglue.config['match_threshold']
            small_text = [
                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                'Match Threshold: {:.2f}'.format(m_thresh),
                'Image Pair: {}:{}'.format(stem0, stem1),
            ]

            #Actually creates and saves the visualization image; Draws keypoints and matches, colored by epipolar error; 
            # Overlays thet text+parameter info; Output is saved at viz_eval_path
            make_matching_plot(
                image0, image1, kpts0, kpts1, mkpts0,
                mkpts1, color, text, viz_eval_path,
                opt.show_keypoints, opt.fast_viz,
                opt.opencv_display, 'Relative Pose', small_text)

            #Updates timing log for performance profiling (viz_eval stage)
            timer.update('viz_eval')

        #Prints progress: how many pairs have been processed so far
        timer.print('Finished pair {:5} of {:5}'.format(i, len(pairs)))

    if opt.eval:
        # Collate the results into a final table and print to terminal.
        pose_errors = []
        precisions = []
        matching_scores = []
        #Iterates over all iamge pairs; Loads the saved evaluation results from each pair
        for pair in pairs:
            name0, name1 = pair[:2]
            stem0, stem1 = Path(name0).stem, Path(name1).stem
            eval_path = output_dir / \
                '{}_{}_evaluation.npz'.format(stem0, stem1)
            results = np.load(eval_path)
            #Defines pose error as the maximum of translation error (error_t) and rotation error (error_R)
            pose_error = np.maximum(results['error_t'], results['error_R'])
            pose_errors.append(pose_error)
            #Precision = fraction of correct matches out of all matches
            precisions.append(results['precision'])
            #Matching score = fraction of correct matches out of all detected keypoints
            matching_scores.append(results['matching_score'])
        #Evaluates pose estimation quality using AUC (Area Under Curve) at threshold
        thresholds = [5, 10, 20]
        aucs = pose_auc(pose_errors, thresholds)
        aucs = [100.*yy for yy in aucs]
        #Computes the average precision and average matching score (as percentage)
        prec = 100.*np.mean(precisions)
        ms = 100.*np.mean(matching_scores)
        #Prints a final results table with the evaluation summary
        print('Evaluation Results (mean over {} pairs):'.format(len(pairs)))
        print('AUC@5\t AUC@10\t AUC@20\t Prec\t MScore\t')
        print('{:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t {:.2f}\t'.format(
            aucs[0], aucs[1], aucs[2], prec, ms))