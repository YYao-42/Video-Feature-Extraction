import numpy as np
import cv2 as cv
import os
import vputils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video folder")
ap.add_argument("-o", "--output", required=True,
	help="path to output video folder")
ap.add_argument("-fps", "--fps", type=int,
	help="fps of videos")
ap.add_argument("-bint", "--blkint", type=int, default=30,
	help="blink interval")
ap.add_argument("-bdur", "--blkdur", type=float, default=0.5,
	help="blink duration")
ap.add_argument("-ext", "--extperc", type=float, default=0.1,
	help="W and H of frames are extended by ext%")
ap.add_argument("-box", "--boxperc", type=float, default=0.1,
	help="side length of the flickering box as the percentage of the original H of the frame")
ap.add_argument("-fdur", "--fadedur", type=int, default=3,
	help="duration of cross fading")
args = vars(ap.parse_args())

video_list = os.listdir(args["input"])
fps = args["fps"]
blink_interval = args["blkint"]
blink_duration = args["blkdur"]
ext_percent = args["extperc"]
box_percent = args["boxperc"]
fade_duration = args["fadedur"]

print("[INFO] Inserting flickering box into videos ...")
for video in video_list:
    print('currently working on: ' + video)
    video_path = args["input"] + video
    write_path = args["output"] + video[:-4] + '_box.avi'
    H_extend, W_extend, box_len = vputils.add_flickering_box(video_path, write_path, fps, blink_interval, blink_duration, ext_percent, box_percent)

video_box_list = [video for video in os.listdir(args["output"]) if video.endswith('_box.avi')]
write_path = args["output"] + 'exp.avi'

print("[INFO] Concatenating videos ...")
# initialize writer
fourcc = cv.VideoWriter_fourcc(*"MJPG")
writer = cv.VideoWriter(write_path, fourcc, fps, (W_extend, H_extend), True)
# append 1s black frames at the beginning
for i in range(fps):
    black = np.zeros([H_extend, W_extend, 3], dtype=np.uint8)
    writer.write(black)
# do cross fading at the boundary of two videos
last_frame = None
for video_box in video_box_list:
    print('currently working on: ' + video_box)
    video_box_path = args["output"] + video_box
    cap = cv.VideoCapture(video_box_path)
    ret, first_frame = cap.read()
    if last_frame is not None:
        frame_list = vputils.cross_fading(last_frame, first_frame, fade_duration, fps, box_len)
        for frame in frame_list:
            writer.write(frame)
    else:
        writer.write(first_frame)
    while (True):
        ret, frame = cap.read()
        if not ret:
            break
        else:
            last_frame = frame
            writer.write(frame)
# append 1s grey frames at the end
for i in range(fps):
    grey = np.zeros([H_extend, W_extend, 3], dtype=np.uint8)
    grey.fill(169)
    writer.write(grey)
writer.release()
cap.release()