# Adapted from
# https://pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# import the necessary packages
import numpy as np
import argparse
import cv2
import os
import feutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-m", "--model", default='ssd',
	help="base path to the directory of the pretrained model")
ap.add_argument('-optic', '--opticalonly', action='store_true',
    help='Include if use optical only (i.e., without face detection)')
ap.add_argument('-GPU', '--GPU', action='store_true',
    help='Include if use GPU acceleration')
ap.add_argument("-nb", "--nbins", type=int, default=8,
	help="number of bins of the histogram")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# derive the paths to the Mask R-CNN weights and model configuration
weightsPath = os.path.sep.join([args["model"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
configPath = os.path.sep.join([args["model"],
	"deploy.prototxt"])
# load our Mask R-CNN trained on the COCO dataset (90 classes)
# from disk
print("[INFO] loading model from disk...")
net = cv2.dnn.readNetFromCaffe(configPath, weightsPath)

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

# First frame
grabbed, frame_prev = vs.read()
hist_mtx = np.zeros((1, args["nbins"]))
box_mtx = np.zeros((1, 4))
mag_mtx = np.zeros((1, 5))

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
		elap_FF = 0
		hist, box_info, mag, frame_OF, elap_OF = feutils.optical_flow_FB(frame, frame_prev, nb_bins=8, GPU=args["GPU"])
	else:
		# Detect objects 
		boxes, confidences, elap_OF = feutils.face_detection(frame, net, args)
		# Compute the optical flow of the most confidenet detected object
		hist, box_info, mag, frame_OF, elap_FF = feutils.optical_flow_face(frame, frame_prev, boxes, confidences, oneobject=True, nb_bins=args["nbins"])
	hist_mtx = np.concatenate((hist_mtx, hist), axis=0)
	box_mtx = np.concatenate((box_mtx, box_info), axis=0)
	mag_mtx = np.concatenate((mag_mtx, mag), axis=0)
	frame_prev = frame
	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame_OF.shape[1], frame_OF.shape[0]), True)
		# some information on processing single frame
		if total > 0:
			print("[INFO] Face detection: single frame took {:.4f} seconds".format(elap_FF))
			print("[INFO] Optical flow: single frame took {:.4f} seconds".format(elap_OF))
			print("[INFO] estimated total time to finish: {:.4f}".format((elap_FF+elap_OF) * total))
	# write the output frame to disk
	writer.write(frame_OF)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
feats = np.concatenate((hist_mtx, mag_mtx, box_mtx), axis=1)
print("[INFO] saving features ...")
np.save('features/histogram.npy', hist_mtx)
np.save('features/box.npy', box_mtx)
np.save('features/mag.npy', mag_mtx)
if args["opticalonly"]:
	save_path = 'features/' + video_id +'_flow.npy'
else:
	save_path = 'features/' + video_id +'_face.npy'
np.save(save_path, feats)
# release the file pointers
print("[INFO] cleaning up ...")
writer.release()
vs.release()