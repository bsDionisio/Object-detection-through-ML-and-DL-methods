front_matter = """
------------------------------------------------------------------------
Online demo for [LoFTR](https://zju3dv.github.io/loftr/).

This demo is heavily inspired by [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork/).
We thank the authors for their execellent work.
------------------------------------------------------------------------
"""

import os
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import matplotlib.cm as cm

from types import SimpleNamespace

os.sys.path.append("../")  # Add the project directory to python's module path
from src.loftr import LoFTR, default_cfg
from src.config.default import get_cfg_defaults #Function that returns a default configuration object
try:
    from demo.utils import (AverageTimer, VideoStreamer,
                            make_matching_plot_fast, make_matching_plot, frame2tensor)
except:
    raise ImportError("This demo requires utils.py from SuperGlue, please use run_demo.sh to start this script.")


torch.set_grad_enabled(False)

if __name__ == '__main__':
    opt = SimpleNamespace(
        weight='/home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/DL_algorithms/LoFTR/weights/outdoor_ds.ckpt',
        input='/home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/data',  # webcam ID, or path to video/images
        output_dir='/home/buybluepants/Documents/Object-detection-through-ML-and-DL-methods/results',  # where to save frames
        image_glob=['frame.png', 'logo.png'],
        skip=1,
        max_length=1000000,
        resize=[640, 480],
        no_display=False,   #If True, will skip cv2.imshow(), only save images/video if configured
        save_video=True,    #outputs a video showing the matches
        save_input=False,   #writes a video of the original frames
        skip_frames=1,
        top_k=500,
        bottom_k=0
    )

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu'

    # Initialize LoFTR
    matcher = LoFTR(config=default_cfg) #Creates a LoFTR model instance with the default configuration
    #Loads the pretrained model weights from a checkpoint file given in opt.weight
    matcher.load_state_dict(torch.load(opt.weight)['state_dict'])
    #Puts the model in evaluation mode and moves the model to the specified device
    matcher = matcher.eval().to(device=device)

    # Configure I/O
    if opt.save_video:
        #Uses OpenCV's VideoWriter to optionally save videos; loftr-matches.mp4 will contain side-by-side matching visualizations;
        print('Writing video to loftr-matches.mp4...')
        writer = cv2.VideoWriter('loftr-matches.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640*2 + 10, 480))
    if opt.save_input:
        #demo-input.mp4 will contain the raw input video frames
        print('Writing video to demo-input.mp4...')
        input_writer = cv2.VideoWriter('demo-input.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (640, 480))

    #Utility for grabbing frames from a webcam, video file, or image sequence
    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    #Returns a tuple (frame, ret); frame=the actual image (numpy array); ret=boolean -> True if reading succeeded
    frame, ret = vs.next_frame()
    #Makes sure the first frame was read correctly; If not, stops with an error
    assert ret, 'Error when reading the first frame (try different --input?)'

    #Counters for tracking frames
    frame_id = 0  
    last_image_id = 0
    #Converts the OpenCV frame (BGR numpy array) into a PyTorch tensor suitable for the model; Moves to correct device
    frame_tensor = frame2tensor(frame, device)
    #Prepares the first frame in the format expected
    last_data = {'image0': frame_tensor}
    #Keeps a copy of the raw image
    last_frame = frame

    #If the user provided --output_dir, then it prints the chosen directory and creates the dictionary using Path.mkdir
    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        #exist_ok=True voids errors if the folder already exists
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        window_name = 'LoFTR Matches'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, (640*2, 480))
    #If visualization is disabled, it prints a message that no GUI will be shown
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the reference image (left)\n'
          '\td/f: move the range of the matches (ranked by confidence) to visualize\n'
          '\tc/v: increase/decrease the length of the visualization range (i.e., total number of matches) to show\n'
          '\tq: quit')

    #starts a utility that tracks average runtime
    timer = AverageTimer()
    #Defines which subset of matches will be shown
    vis_range = [opt.bottom_k, opt.top_k]

    #Loop forever until frames run out
    while True:
        #Counts processed frames
        frame_id += 1
        #Grabs the next frame from input
        frame, ret = vs.next_frame()    #ret= True if read worked, False if no more frames
        #If --skip_frames >1, only processes every Nth frames (saves compute)
        if frame_id % opt.skip_frames != 0:
            # print("Skipping frame.")
            #Skipped frames are ignored
            continue
        #if --save_input was set
        if opt.save_input:
            inp = np.stack([frame]*3, -1)
            #Converts grayscale frames into RGB (needed because video writers expect 3 channels)
            inp_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            #Writes the frame into demo-input.mp4
            input_writer.write(inp_rgb)
        #If no frame ws read (ret==False), the loop ends and the demo stops
        if not ret:
            print('Finished demo_loftr.py')
            break
        #Records the time spent on data loading
        timer.update('data')
        #Identifiers for the reference frame (last_image_id) and current frame (vs.i-1)
        stem0, stem1 = last_image_id, vs.i - 1

    #Converts current frame -> PyTorch tensor
        frame_tensor = frame2tensor(frame, device)
        #Updates last_data: image0-reference frame (from before); image1: current frame (new)
        last_data = {**last_data, 'image1': frame_tensor}
        #This fills last_data with: mkpts0_f: matched keypoints in reference frame; mkpts1_f=matched keypoints in current frame;
        # mconf=confidence scores or each match
        matcher(last_data)

        #total number of matches LoFTR found
        total_n_matches = len(last_data['mkpts0_f'])
        #reference frame keypoints (subset defined by vis_range)
        mkpts0 = last_data['mkpts0_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        #current frame keypoints (same indices)
        mkpts1 = last_data['mkpts1_f'].cpu().numpy()[vis_range[0]:vis_range[1]]
        #Confidence score for those matches
        mconf = last_data['mconf'].cpu().numpy()[vis_range[0]:vis_range[1]]

        # Normalize confidence. Scales confidence scores into [0,1] for consistent coloring. Prevents division by zero with + 1e-5
        if len(mconf) > 0:
            conf_vis_min = 0.
            conf_min = mconf.min()
            conf_max = mconf.max()
            mconf = (mconf - conf_vis_min) / (conf_max - conf_vis_min + 1e-5)

        # 🔥 Filter low-confidence matches
        conf_threshold = 0.7  # try 0.3, 0.5, 0.7
        keep = mconf > conf_threshold
        mkpts0 = mkpts0[keep]
        mkpts1 = mkpts1[keep]
        mconf  = mconf[keep]

        #Records model inference time
        timer.update('forward')
        #sets transparency level (no transparency here)
        alpha = 0
        #cm.jet(): maps normalized confidence scores into jet colormap (blue→green→yellow→red)
        color = cm.jet(mconf, alpha=alpha)

        #Large caption that appears at the top of the visualization
        text = [
            f'LoFTR',
            '# Matches (showing/total): {}/{}'.format(len(mkpts0), total_n_matches),
        ]
        #Smaller annotation, sually shown at the bottom or corner of the plot
        small_text = [
            f'Showing matches from {vis_range[0]}:{vis_range[1]}',
            f'Confidence Range: {conf_min:.2f}:{conf_max:.2f}',
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        #Calls make_matching_plot_fast (a helper function from demo/utils.py); 
        # Output: out=a single OpenCV image showing both frames side by side with lines connecting matched points.
        out = make_matching_plot_fast(
            last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=False, small_text=small_text)

        # Save high quality png, optionally with dynamic alpha support (unreleased yet).
        # save_path = 'demo_vid/{:06}'.format(frame_id)
        # make_matching_plot(
        #     last_frame, frame, mkpts0, mkpts1, mkpts0, mkpts1, color, text,
        #     path=save_path, show_keypoints=opt.show_keypoints, small_text=small_text)

        if not opt.no_display:
            if opt.save_video:
                writer.write(out)   #appends the visualization to the output video if --save_video
            cv2.imshow('LoFTR Matches', out)    #displays the match visualization window
            key = chr(cv2.waitKey(1) & 0xFF)    #cv2.waitKey(1) → waits for a keypress (1 ms); Converts to character
            #Quit (q)
            if key == 'q':
                #Releases video writers
                if opt.save_video:
                    writer.release()
                if opt.save_input:
                    input_writer.release()
                #Cleans up input stream
                vs.cleanup()
                print('Exiting...')
                #Exists the loop
                break
            #New reference image
            elif key == 'n':  
                #Sets the current frame as the new reference (image0); Future matches will be compared against this frame
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
                frame_id_left = frame_id
            #Scroll Through Match Range (d/f)
            elif key in ['d', 'f']:
                #Moves the window of matches shown (by +/- 200)
                if key == 'd':
                    if vis_range[0] >= 0:
                       vis_range[0] -= 200
                       vis_range[1] -= 200
                if key =='f':
                    vis_range[0] += 200
                    vis_range[1] += 200
                print(f'\nChanged the vis_range to {vis_range[0]}:{vis_range[1]}')
            #Shrinks or expands how many matches are shown un the visualization by +/- 50
            elif key in ['c', 'v']:
                if key == 'c':
                    vis_range[1] -= 50
                if key =='v':
                    vis_range[1] += 50
                print(f'\nChanged the vis_range[1] to {vis_range[1]}')
        #If no GUI dislay (--no_display) -> Saves each frame's visualization as a .png in the output directory
        elif opt.output_dir is not None:
            stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            out_file = str(Path(opt.output_dir, stem + '.png'))
            print('\nWriting image to {}'.format(out_file))
            cv2.imwrite(out_file, out)
        #If neither display nor output directory is given -> error, since results would be lost
        else:
            raise ValueError("output_dir is required when no display is given.")
        #Updates the visualization time statistics; prints FPS/per-stage timing (dara, forward, viz)
        timer.update('viz')
        timer.print()


    #Closes all OpenCV windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Cleans up the VideoStreamer (releases video file or camera)
    vs.cleanup()