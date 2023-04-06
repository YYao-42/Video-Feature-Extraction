# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import feutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video folder")
args = vars(ap.parse_args())

if not args["input"].endswith('/'):
    args["input"] = args["input"] + '/'
# list videos in the input folder
video_list = [f for f in os.listdir(args["input"]) if f.endswith('.avi') or f.endswith('.mp4')] # list all video files
for video in video_list:
    print("[INFO] Working on " + video + " ...")
    video_id = video[:2]
    video_path = args["input"] + video
    tempcontr_mtx = feutils.get_tempcontrast(video_path)
    np.save('features/' + video_id + '_tempctra.npy', tempcontr_mtx)