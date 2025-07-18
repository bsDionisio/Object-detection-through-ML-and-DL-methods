import sys
import json
import cv2
import numpy as np

def main():
    # Read input from stdin
    input_line = sys.stdin.readline()
    if not input_line:
        return

    # Parse the input JSON
    input_data = json.loads(input_line)

    # Set the default video URL if videoURL is null or empty
    video_url = input_data.get('videoURL', 'rtsp://127.0.0.1:8554/gcs_in')

    # Open the video stream
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print(json.dumps({'error': 'Cannot open video stream'}))
        sys.stdout.flush()
        return

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print(json.dumps({'error': 'Cannot read video frame'}))
        sys.stdout.flush()
        return

    # Get the frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # Get percentages from input data
    x_percent = input_data.get('xPercent', 0.5)
    y_percent = input_data.get('yPercent', 0.5)
    window_width_percent = input_data.get('windowWidthPercent', 0.05)
    window_height_percent = input_data.get('windowHeightPercent', 0.05)

    # Calculate pixel coordinates and dimensions
    window_width = int(window_width_percent * frame_width)
    window_height = int(window_height_percent * frame_height)
    x = int(x_percent * frame_width) - window_width // 2
    y = int(y_percent * frame_height) - window_height // 2

    # Ensure coordinates are within frame boundaries
    x = max(0, min(x, frame_width - window_width))
    y = max(0, min(y, frame_height - window_height))

    # Define the initial bounding box
    bbox = (x, y, window_width, window_height)

    # Initialize the tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Initialize ORB detector and BFMatcher for re-detection
    orb = cv2.ORB_create(nfeatures=1000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Variables to store keypoints and descriptors
    keypoints_initial = None
    descriptors_initial = None

    # Re-detection variables
    frames_since_lost = 0
    re_detection_delay = 30  # Number of frames to wait before attempting re-detection

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update the tracker
        success, bbox = tracker.update(frame)
        frame_height, frame_width = frame.shape[:2]

        if success:
            # Tracking is successful
            frames_since_lost = 0  # Reset the frames since lost

            # Ensure the bounding box is valid (within frame dimensions)
            x, y, w, h = [int(v) for v in bbox]
            x = max(0, min(x, frame_width - w))
            y = max(0, min(y, frame_height - h))
            w = max(1, min(w, frame_width - x))
            h = max(1, min(h, frame_height - y))

            # Extract ROI for updating descriptors
            roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            keypoints_initial, descriptors_initial = orb.detectAndCompute(gray_roi, None)
            if keypoints_initial is not None:
                # Adjust keypoints to the frame coordinates
                for kp in keypoints_initial:
                    kp.pt = (kp.pt[0] + x, kp.pt[1] + y)

            tracking_data = {
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'frameWidth': frame_width,
                'frameHeight': frame_height
            }

            # Send tracking data back
            print(json.dumps(tracking_data))
            sys.stdout.flush()
        else:
            # Tracker failed
            frames_since_lost += 1

            if frames_since_lost >= re_detection_delay and descriptors_initial is not None and len(descriptors_initial) > 0:
                # Attempt to re-detect the object
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                keypoints_frame, descriptors_frame = orb.detectAndCompute(gray_frame, None)

                if descriptors_frame is not None and len(descriptors_frame) > 0:
                    # Match descriptors between initial and current frame
                    matches = bf.match(descriptors_initial, descriptors_frame)
                    matches = sorted(matches, key=lambda x: x.distance)
                    good_matches = matches[:50]  # Adjust the number as needed

                    if len(good_matches) >= 10:
                        src_pts = np.float32([keypoints_initial[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([keypoints_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Compute homography to find new bounding box location
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        if M is not None:
                            h_roi, w_roi = gray_roi.shape[:2]
                            pts = np.float32([[0, 0], [w_roi, 0], [w_roi, h_roi], [0, h_roi]]).reshape(-1, 1, 2)
                            dst = cv2.perspectiveTransform(pts, M)

                            # Get new bounding box from transformed points
                            x_new, y_new, w_new, h_new = cv2.boundingRect(dst)

                            # Ensure coordinates are within frame boundaries
                            x_new = max(0, min(int(x_new), frame_width - 1))
                            y_new = max(0, min(int(y_new), frame_height - 1))
                            w_new = max(1, min(int(w_new), frame_width - x_new))
                            h_new = max(1, min(int(h_new), frame_height - y_new))

                            # Re-initialize the tracker with the new bounding box
                            bbox = (x_new, y_new, w_new, h_new)
                            tracker = cv2.TrackerCSRT_create()
                            tracker.init(frame, bbox)
                            frames_since_lost = 0  # Reset frames since lost

                            # Update the tracking data
                            tracking_data = {
                                'x': x_new,
                                'y': y_new,
                                'w': w_new,
                                'h': h_new,
                                'frameWidth': frame_width,
                                'frameHeight': frame_height
                            }
                            print(json.dumps(tracking_data))
                            sys.stdout.flush()
                            continue  # Continue to next frame
            else:
                # Output zero tracking data
                tracking_data = {
                    'x': 0,
                    'y': 0,
                    'w': 0,
                    'h': 0,
                    'frameWidth': frame_width,
                    'frameHeight': frame_height
                }
                print(json.dumps(tracking_data))
                sys.stdout.flush()

        # Optional: add a delay if needed
        # cv2.waitKey(1)

    cap.release()

if __name__ == '__main__':
    main()

