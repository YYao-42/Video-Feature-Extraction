import numpy as np
import cv2 as cv

def extend_frame(H_extend, W_extend, frame, black, box_len):
    H, W = frame.shape[:2]
    W_pad = np.zeros([H, W_extend-W, 3], dtype=np.uint8)
    frame_extend_W = np.concatenate((frame, W_pad), axis=1)
    H_pad = np.zeros([H_extend-H, W_extend, 3], dtype=np.uint8)
    if not black:
        H_pad[:box_len, -box_len:, :] = 255
    frame_extended = np.concatenate((H_pad, frame_extend_W), axis=0)
    return frame_extended


def add_flickering_box(video_path, write_path, fps, blink_interval, blink_duration, ext_percent=0.1, box_percent=0.1):
    # initialize the video stream
    cap = cv.VideoCapture(video_path)
    # ret = a boolean return value from getting the frame, frame = the first frame in the entire video sequence
    ret, frame = cap.read()
    H, W = frame.shape[:2]
    H_extend = int(H*(1+ext_percent))
    W_extend = int(W*(1+ext_percent))
    box_len = int(H*box_percent)
    # initialize writer
    fourcc = cv.VideoWriter_fourcc(*"MJPG")
    writer = cv.VideoWriter(write_path, fourcc, fps, (W_extend, H_extend), True)
    # add a black strip on the top of each frame
    idx_frame = 0
    while (ret):
        # a white box will occur for blink_duration seconds per blink_interval seconds
        if (idx_frame % (fps*blink_interval)) == 0:
            flag_black = 0
            countdown = blink_duration*fps
        extended_frame = extend_frame(H_extend, W_extend, frame, black=flag_black, box_len=box_len)
        if not flag_black:
            countdown = countdown - 1
            if countdown == 0:
                flag_black = 1
        writer.write(extended_frame)
        ret, frame = cap.read()
        idx_frame = idx_frame + 1
    writer.release()
    cap.release()
    return H_extend, W_extend, box_len


def cross_fading(last_frame, first_frame, duration, fps, box_len):
    frame_list = []
    n_points = duration*fps
    percentage = np.linspace(0, 1, n_points)
    for perc in percentage:
        frame = cv.addWeighted(last_frame, 1-perc, first_frame, perc, 0)
        frame[:box_len, -box_len:, :] = 169
        frame_list.append(frame)
    return frame_list