# Adapted from
# https://pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
# https://github.com/pacocp/YOLOF

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-dl", "--detectlabel", type=str, default='person',
	help="class of objects to be detected")
ap.add_argument("-nb", "--nbins", type=int, default=8,
	help="number of bins of the histogram")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov4.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov4.cfg"])
# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames() # layer names
# Only include layers that are not connected by follow up layers
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)
# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# First frame
grabbed, frame_prev = vs.read()
hist_mtx = np.zeros((1, args["nbins"]))
center_mtx = np.zeros((1, 2))

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	grabbed, frame = vs.read()
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		H, W = frame.shape[:2]
	# Detect objects 
	boxes, confidences, classIDs, idxs, elap_OD = utils.object_detection_yolo(frame, net, ln, W, H, args, LABELS, detect_label=args["detectlabel"])
	# Compute the optical flow of the most confidenet detected object
	hist, center_xy, frame_OF, elap_OF = utils.optical_flow_FB(frame, frame_prev, boxes, confidences, classIDs, idxs, LABELS, COLORS, oneobject=True, nb_bins=8)
	hist_mtx = np.concatenate((hist_mtx, hist), axis=0)
	center_mtx = np.concatenate((center_mtx, center_xy), axis=0)
	frame_prev = frame
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame_OF.shape[1], frame_OF.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			print("[INFO] Object detection: single frame took {:.4f} seconds".format(elap_OD))
			print("[INFO] Optical flow: single frame took {:.4f} seconds".format(elap_OF))
			print("[INFO] estimated total time to finish: {:.4f}".format((elap_OD+elap_OF) * total))
			# print("[INFO] estimated total time to finish: {:.4f}".format((elap_OD) * total))
	# write the output frame to disk
	writer.write(frame_OF)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
print("[INFO] saving features ...")
np.save('features/histogram.npy', hist_mtx)
np.save('features/center.npy', center_mtx)
# release the file pointers
print("[INFO] cleaning up ...")
writer.release()
vs.release()