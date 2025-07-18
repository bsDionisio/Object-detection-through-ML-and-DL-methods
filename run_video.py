import cv2 as cv2
# from my_sift import Sift
from my_sift_light import SiftLight
# from my_orb import Orb
from my_DL import DL

was_clicked = False

x_tl = None
y_tl = None
x_br = None
y_br = None


def click_event(event, x, y, flags, param):
    global was_clicked
    global x_tl, y_tl, x_br, y_br
    print(event, x, y, flags, param, was_clicked)
    if flags == 32:  # CONTROL KEY
        x_tl, y_tl, x_br, y_br = None, None, None, None
        was_clicked = False
        my_sift_object.img1 = None
    if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_SHIFTKEY + 1:
        print(f"Click coordenates: x = {x}, y = {y}, flags = {flags}, param = {param}")
        if x_tl is None:
            x_tl, y_tl = x, y
            was_clicked = True
        else:
            x_br, y_br = x, y
            was_clicked = False
            # print(f"Top left: {x_tl, y_tl}, Bottom right: {x_br, y_br}")

            f_x_tl = x_tl if x_tl < x_br else x_br
            f_y_tl = y_tl if y_tl < y_br else y_br
            f_x_br = x_br if x_tl < x_br else x_tl
            f_y_br = y_br if y_tl < y_br else y_tl

            my_sift_object.find_key_points_logo(frame[f_y_tl:f_y_br, f_x_tl:f_x_br, :])

            # save the image
            # cv2.imwrite('data/logo.png', frame[f_y_tl:f_y_br, f_x_tl:f_x_br, :])
            # cv2.imwrite('data/frame.png', frame)

            x_tl, y_tl, x_br, y_br = None, None, None, None
    elif event == cv2.EVENT_MOUSEMOVE and was_clicked:
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x_tl, y_tl), (x, y), (0, 255, 0), 2)
        cv2.imshow('frame', frame_copy)


# my_sift_object = Orb()
# my_sift_object = Sift()
my_sift_object = SiftLight()
# my_sift_object = DL()
cap = cv2.VideoCapture('data/vtol_repmus_cut.mp4')
#width = 640
#height = 480
while True:
    if not was_clicked:
        ret, frame = cap.read()
        frame = frame[:, :frame.shape[1]//2, :]
        # frame = cv2.resize(frame, (width, height))

        if my_sift_object.img1 is not None:
            scene_corners = my_sift_object.find_matches(frame)
            if scene_corners is not None:
                res_img = frame.copy()
                cv2.line(res_img, (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])),
                         (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
                cv2.line(res_img, (int(scene_corners[1, 0, 0]), int(scene_corners[1, 0, 1])),
                         (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
                cv2.line(res_img, (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1])),
                         (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
                cv2.line(res_img, (int(scene_corners[3, 0, 0]), int(scene_corners[3, 0, 1])),
                         (int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)
            # my_sift_object.img1 = None
                cv2.imshow('frame', res_img)
            else:
                cv2.imshow('frame', frame)
        else:
            cv2.imshow('frame', frame)

    cv2.setMouseCallback('frame', click_event)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
