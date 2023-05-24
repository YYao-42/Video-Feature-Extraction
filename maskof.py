# Adapted from
# https://pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import os
import feutils
import pickle

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-m", "--mask-rcnn", default='mask-rcnn',
	help="base path to mask-rcnn directory")
ap.add_argument('-optic', '--opticalonly', action='store_true',
    help='Include if use optical only (i.e., without object detection)')
ap.add_argument('-GPU', '--GPU', action='store_true',
    help='Include if use GPU acceleration')
ap.add_argument("-dl", "--detectlabel", type=str, default='person',
	help="class of objects to be detected")
ap.add_argument("-nb", "--nbins", type=int, default=8,
	help="number of bins of the histogram")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,
	help="minimum threshold for pixel-wise mask segmentation")
args = vars(ap.parse_args())

# load the COCO class labels our Mask R-CNN was trained on
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# load the set of colors that will be used when visualizing a given
# instance segmentation
# colorsPath = os.path.sep.join([args["mask_rcnn"], "colors.txt"])
# COLORS = open(colorsPath).read().strip().split("\n")
# COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
# COLORS = np.array(COLORS, dtype="uint8")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])
# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

if args["GPU"]:
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# initialize the video stream, pointer to output video file, and
# frame dimensions
video_path = args["input"]
video_name = video_path.split('/')[-1]
video_id = video_name.split('_')[0]
vs = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)
# try to determine the total number of frames in the video file
try:
	total = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
	print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

feature_list = []

# First frame
grabbed, frame_prev = vs.read()
hist_mtx = np.zeros((1, args["nbins"]))
box_mtx = np.zeros((1, 4))
mag_mtx = np.zeros((1, 5))
feat_1st = np.zeros((1, args["nbins"]+4+5))
feature_list.append(feat_1st)

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
	if args["opticalonly"]:
		elap_OS = 0
		hist, box_info, mag, frame_OF, elap_OF = feutils.optical_flow_FB(frame, frame_prev, nb_bins=8, GPU=args["GPU"])
		hist_mtx = np.concatenate((hist_mtx, hist), axis=0)
		box_mtx = np.concatenate((box_mtx, box_info), axis=0)
		mag_mtx = np.concatenate((mag_mtx, mag), axis=0)
	else:
		# Detect objects 
		boxes, confidences, classIDs, masks, elap_OS = feutils.object_seg_maskrcnn(frame, net, args, LABELS, detect_label=args["detectlabel"])
		# Compute the optical flow of the most confidenet detected object
		feature_boxes, frame_OF, elap_OF = feutils.optical_flow_mask(frame, frame_prev, boxes, confidences, classIDs, masks, LABELS, COLORS, oneobject=False, nb_bins=8)
		feature_list.append(feature_boxes)
	frame_prev = frame
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame_OF.shape[1], frame_OF.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			print("[INFO] Object segmentation: single frame took {:.4f} seconds".format(elap_OS))
			print("[INFO] Optical flow: single frame took {:.4f} seconds".format(elap_OF))
			print("[INFO] estimated total time to finish: {:.4f}".format((elap_OS+elap_OF) * total))
	# write the output frame to disk
	writer.write(frame_OF)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
print("[INFO] saving features ...")
if args["opticalonly"]:
	save_path = 'features/' + video_id +'_flow.npy'
	feats = np.concatenate((hist_mtx, mag_mtx, box_mtx), axis=1)
	np.save(save_path, feats)
else:
	save_path = 'features/' + video_id +'_mask.pkl'
	open_file = open(save_path, "wb")
	pickle.dump(feature_list, open_file)
	open_file.close()
# release the file pointers
print("[INFO] cleaning up ...")
writer.release()
vs.release()