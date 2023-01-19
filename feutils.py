import numpy as np
import cv2 as cv
import time
import copy


def HOOF(magnitude, angle, nb_bins, fuzzy=False, normalize=False):
	'''
	Histogram of (fuzzy) oriented optical flow
	Inputs:
	magnitude: magnitude matrix of the optical flow
	angle: angle matrix of the optical flow
	nb_bins: number of bins
	fuzzy: whether include fuzzy matrix https://ieeexplore.ieee.org/document/7971947/. Default is False.
	nomalize: whether normalize the histogram to be a pdf. Default is False such that we don't lose information (sum of the magnitude), 
	          which gives us more freedom on post-processing (like smoothing).
	Outputs:
	hist: normalized and weighted orientation histogram with size 1 x nb_bins
	Note: The normalized histogram does not sum to 1; instead, np.sum(hist)*2pi/nb_bins = 1
	'''
	# Flatten mag/ang matrices
	magnitude = magnitude.flatten()
	angle = angle.flatten()
	# Deal with circular continuity
	for i in range(len(angle)):
		while (angle[i] < 0 or angle[i] > 2*np.pi):
			angle[i] = angle[i] - np.sign(angle[i])*2*np.pi
	# Normalized histogram weighted by magnitudes
	if fuzzy:
		x = np.linspace(0, 2*np.pi, nb_bins*2+1)
		bin_mid = x[[list(range(1, 2*nb_bins, 2))]]
		nb_bins_dense = nb_bins*5
		x = np.linspace(0, 2*np.pi, nb_bins_dense*2+1)
		bin_dense_mid = x[[list(range(1, 2*nb_bins_dense, 2))]]
		diff_mtx = np.minimum(np.abs(bin_mid-bin_dense_mid.T), 2*np.pi-np.abs(bin_mid-bin_dense_mid.T))
		sigma = 0.1 # May not be the best value
		coe_mtx = np.exp(-diff_mtx**2/2/sigma**2) # fuzzy matrix
		hist_dense, _ = np.histogram(angle, nb_bins_dense, range=(0, 2*np.pi), weights=magnitude, density=False)
		hist = np.expand_dims(hist_dense, axis=0)@coe_mtx
		hist = hist/np.sum(hist)/2/np.pi*nb_bins
	else:
		hist, _ = np.histogram(angle, nb_bins, range=(0, 2*np.pi), weights=magnitude, density=normalize)
		hist = np.expand_dims(hist, axis=0)
	return hist


def object_detection_yolo(frame, net, ln, W, H, args, LABELS, detect_label='person'):
	'''
	Inputs:
	frame: current frame
	net: yolo net for object detection
	ln: layer names of layers to be retrieved
	W: width of the frame
	H: height of the frame
	args: pre-defined parameters
	LABELS: labels of all classes
	detect_label: class to be detected; Default: person
	Outputs:
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	idxs: indices of refined boxes 
	elap: processing time
	'''
	start = time.time()
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)
	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []
	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"] and LABELS[classID]==detect_label:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])
	end = time.time()
	elap = end-start
	return boxes, confidences, classIDs, idxs, elap


def optical_flow_FB(frame, frame_prev, nb_bins=8):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	nb_bins: number of bins (for Histogram of Oriented Optical Flow)
	Outputs:
	hist: orientation histogram
	center_xy: NaN (Keep here just to have the same number of outputs as optical_flow_box)
	mag: average magnitude (all direction/left/right/up/down) 
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	flow = cv.calcOpticalFlowFarneback(frame_prev_grey, frame_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	flow_horizontal = flow[..., 0]
	flow_vertical = flow[..., 1]
	# Computes the magnitude and angle of the 2D vectors
	# DON'T TRUST cv.cartToPolar
	# magnitude, angle = cv.cartToPolar(flow_horizontal, flow_vertical, angleInDegrees=False)
	magnitude = np.absolute(flow_horizontal+1j*flow_vertical)
	angle = np.angle(flow_horizontal+1j*flow_vertical)
	mag = []
	mag.append([
		magnitude.mean(), # avg magnitude
		flow_horizontal[flow_horizontal >= 0].mean(),  # up
		flow_horizontal[flow_horizontal <= 0].mean(),  # down
		flow_vertical[flow_vertical <= 0].mean(),  # left
		flow_vertical[flow_vertical >= 0].mean()  # right
	])
	mag = np.asarray(mag)
	hist = HOOF(magnitude, angle, nb_bins, fuzzy=False, normalize=False)
	center_xy = np.full((1, 2), np.nan)
	hsv = np.zeros_like(frame)
	hsv[..., 0] = angle*180/np.pi/2
	hsv[..., 1] = 255
	hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
	frame_OF = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
	cv.imshow("object detection + optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return hist, center_xy, mag, frame_OF, elap


def optical_flow_box(frame, frame_prev, boxes, confidences, classIDs, idxs, LABELS, COLORS, oneobject=True, nb_bins=8):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	idxs: indices of refined boxes
	LABELS: all labels
	COLORS: colors assigned to labels
	oneobject: if only select one object with highest confidence
	Outputs:
	hist: orientation histogram
	center_xy: x and y coordinates of the center of the box
	mag: average magnitude (all direction/left/right/up/down) 
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_OF = copy.deepcopy(frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	if len(idxs)==0:
		print('WARNING: No object detected! Adding NaN values to features.')
		hist = np.full((1, nb_bins), np.nan)
		center_xy = np.full((1, 2), np.nan)
		mag = np.full((1, 5), np.nan)
	else:
		idxs = idxs.flatten()
		if oneobject:
			idx_maxconfi = np.argmax(np.array(confidences)[idxs])
			idxs = [idxs[idx_maxconfi]]
		# TODO: save features for multiobjects
		for i in idxs:
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			center_xy = np.expand_dims(np.array([x+w/2, y+h/2]), axis=0)
			ys = max(0,y)
			yl = min(y+h, frame.shape[0])
			xs = max(0, x)
			xl = min(x+w, frame.shape[1])
			patch = frame_grey[ys:yl, xs:xl]
			patch_prev = frame_prev_grey[ys:yl, xs:xl]
			try:
				flow_patch = cv.calcOpticalFlowFarneback(patch_prev, patch, None, 0.5, 3, 15, 3, 5, 1.2, 0)
			except:
				print('WARNING: empty patches!')
				break
			flow_horizontal = flow_patch[..., 0]
			flow_vertical = flow_patch[..., 1]
			# Computes the magnitude and angle of the 2D vectors
			# DON'T TRUST cv.cartToPolar
			# magnitude, angle = cv.cartToPolar(flow_horizontal, flow_vertical, angleInDegrees=False)
			magnitude = np.absolute(flow_horizontal+1j*flow_vertical)
			angle = np.angle(flow_horizontal+1j*flow_vertical)
			if magnitude.mean() > 1e200:
				print("ABNORMAL!")
			mag = []
			mag.append([
				magnitude.mean(), # avg magnitude
				flow_horizontal[flow_horizontal >= 0].mean(),  # up
				flow_horizontal[flow_horizontal <= 0].mean(),  # down
				flow_vertical[flow_vertical <= 0].mean(),  # left
				flow_vertical[flow_vertical >= 0].mean()  # right
			])
			mag = np.asarray(mag)
			hist = HOOF(magnitude, angle, nb_bins, fuzzy=False, normalize=False)
			hsv = np.zeros_like(frame[ys:yl, xs:xl, :])
			hsv[..., 0] = angle*180/np.pi/2
			hsv[..., 1] = 255
			hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
			bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
			frame_OF[ys:yl, xs:xl, :] = bgr
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],	confidences[i])
			color = [int(c) for c in COLORS[classIDs[i]]]
			cv.putText(frame_OF, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv.imshow("object detection + optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return hist, center_xy, mag, frame_OF, elap