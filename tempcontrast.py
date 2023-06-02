# import the necessary packages
import numpy as np
import argparse
import cv2 as cv
import os
import feutils
import pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input video folder")
ap.add_argument("-m", "--mask-rcnn", default='mask-rcnn',
	help="base path to mask-rcnn directory")
ap.add_argument('-tc', '--tconly', action='store_true',
    help='Include if use temporal contrast only (i.e., without object detection)')
ap.add_argument("-dl", "--detectlabel", type=str, default='person',
	help="class of objects to be detected")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv.dnn.readNetFromTensorflow(weightsPath, configPath)

if not args["input"].endswith('/'):
    args["input"] = args["input"] + '/'
# list videos in the input folder
video_list = [f for f in os.listdir(args["input"]) if f.endswith('.avi') or f.endswith('.mp4')] # list all video files
for video in video_list:
    print("[INFO] Working on " + video + " ...")
    video_id = video[:2]
    video_path = args["input"] + video
    if args["tconly"]:
        tempcontr_mtx = feutils.get_tempcontrast(video_path)
        np.save('features/' + video_id + '_tempctra.npy', tempcontr_mtx)
    else:
        # First frame
        vs = cv.VideoCapture(video_path)
        feature_list = []
        # First frame
        grabbed, frame_prev = vs.read()
        feat_1st = np.zeros((1, 3))
        feature_list.append(feat_1st)
        # loop over frames from the video file stream
        while True:
            # read the next frame from the file
            grabbed, frame = vs.read()
            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            # Detect objects 
            boxes, confidences, classIDs, masks, elap_OS = feutils.object_seg_maskrcnn(frame, net, args, LABELS, detect_label=args["detectlabel"])
            feature_boxes, elap_TC = feutils.temp_contrast_box(frame, frame_prev, boxes, confidences, masks, oneobject=True, ratio=2)
            feature_list.append(feature_boxes)
            frame_prev = frame
        save_path = 'features/' + video_id +'_tcbox.pkl'
        open_file = open(save_path, "wb")
        pickle.dump(feature_list, open_file)
        open_file.close()
        vs.release()
