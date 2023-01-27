import numpy as np
import cv2 as cv
import time
import copy
import math


def expand_box(box, mask, frameW, frameH, ratio=2):
	mask_frame = (np.zeros((frameH, frameW))).astype(bool)
	(x, y) = (box[0], box[1])
	(w, h) = (box[2], box[3])
	mask_frame[y:y+h, x:x+w] = mask
	center_x = x + w/2
	center_y = y + h/2
	w_new = int(math.sqrt(ratio)*w)
	h_new = int(math.sqrt(ratio)*h)
	start_x = max(0, math.floor(center_x-w_new/2))
	start_y = max(0, math.floor(center_y-h_new/2))
	end_x = min(frameW, math.ceil(center_x+w_new/2))
	end_y = min(frameH, math.ceil(center_y+h_new/2))
	mask_expand = mask_frame[start_y:end_y, start_x:end_x]
	return start_x, start_y, end_x, end_y, mask_expand
	

def HOOF(magnitude, angle, nb_bins, mask=None, fuzzy=False, normalize=False):
	'''
	Histogram of (fuzzy) oriented optical flow
	Inputs:
	magnitude: magnitude matrix of the optical flow
	angle: angle matrix of the optical flow
	nb_bins: number of bins
	mask: the mask obtained by mask r-cnn
	fuzzy: whether include fuzzy matrix https://ieeexplore.ieee.org/document/7971947/. Default is False.
	nomalize: whether normalize the histogram to be a pdf. Default is False such that we don't lose information (sum of the magnitude), 
	          which gives us more freedom on post-processing (like smoothing).
	Outputs:
	hist: normalized and weighted orientation histogram with size 1 x nb_bins
	Note: The normalized histogram does not sum to 1; instead, np.sum(hist)*2pi/nb_bins = 1
	'''
	if mask is not None:
		magnitude = magnitude[mask]
		angle = angle[mask]
	else:
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


def object_seg_maskrcnn(frame, net, args, LABELS, detect_label='person'):
	'''
	Use pretrained Mask R-CNN network to perform object segmentation
	Inputs:
	frame: current frame
	net: mask r-cnn net for object segmentation
	args: pre-defined parameters
	LABELS: labels of all classes
	detect_label: class to be detected; Default: person
	Outputs:
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	masks: masks of the object
	elap: processing time
	'''
	start = time.time()
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
	net.setInput(blob)
	(boxes_info, masks_info) = net.forward(["detection_out_final",
		"detection_masks"])
	# initialize our lists of detected bounding boxes, confidences,
	# and masks, respectively
	boxes = []
	confidences = []
	classIDs = []
	masks = []
	# loop over the number of detected objects
	for i in range(0, boxes_info.shape[2]):
		# extract the class ID of the detection along with the
		# confidence (i.e., probability) associated with the
		# prediction
		classID = int(boxes_info[0, 0, i, 1])
		confidence = boxes_info[0, 0, i, 2]
		# filter out weak predictions by ensuring the detected
		# probability is greater than the minimum probability
		if confidence > args["confidence"] and LABELS[classID]==detect_label:
			# scale the bounding box coordinates back relative to the
			# size of the frame and then compute the width and the
			# height of the bounding box
			(H, W) = frame.shape[:2]
			box = boxes_info[0, 0, i, 3:7] * np.array([W, H, W, H])
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY
			# extract the pixel-wise segmentation for the object,
			# resize the mask such that it's the same dimensions of
			# the bounding box, and then finally threshold to create
			# a *binary* mask
			mask = masks_info[i, classID]
			mask = cv.resize(mask, (boxW, boxH),
				interpolation=cv.INTER_NEAREST)
			mask = (mask > args["threshold"])
			# extract the ROI of the image but *only* extracted the
			# masked region of the ROI
			# roi = frame[startY:endY, startX:endX][mask]
			boxes.append([startX, startY, boxW, boxH])
			confidences.append(float(confidence))
			classIDs.append(classID)
			masks.append(mask)

	end = time.time()
	elap = end-start
	return boxes, confidences, classIDs, masks, elap


def optical_flow_FB(frame, frame_prev, nb_bins=8, GPU=False):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	nb_bins: number of bins (for Histogram of Oriented Optical Flow)
	Outputs:
	hist: orientation histogram
	box_info: NaN (Keep here just to have the same number of outputs as optical_flow_box)
	mag: average magnitude (all direction/left/right/up/down) 
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	if GPU:
		frame_grey_gpu = cv.cuda_GpuMat(frame_grey)
		frame_prev_grey_gpu = cv.cuda_GpuMat(frame_prev_grey)
		optical_flow_gpu = cv.cuda.FarnebackOpticalFlow_create(3, 0.5, False, 15, 3, 5, 1.2, 0)
		flow = optical_flow_gpu.calc(frame_prev_grey_gpu, frame_grey_gpu, None)
	else:
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
	box_info = np.full((1, 4), np.nan)
	hsv = np.zeros_like(frame)
	hsv[..., 0] = angle*180/np.pi/2
	hsv[..., 1] = 255
	hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
	frame_OF = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
	cv.imshow("object detection + optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return hist, box_info, mag, frame_OF, elap


def optical_flow_box(frame, frame_prev, boxes, confidences, classIDs, idxs, LABELS, COLORS, oneobject=True, nb_bins=8, expand=False):
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
	box_info: x and y coordinates of the center of the box, width and height of the box
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
		box_info = np.full((1, 4), np.nan)
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
			center_x = x + w/2
			center_y = y + h/2 
			if expand:
				w = int(math.sqrt(3)*w)
				h = int(math.sqrt(3)*h)
				x = int(center_x-w/2)
				y = int(center_y-h/2)
			box_info = np.expand_dims(np.array([center_x, center_y, w, h]), axis=0)
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
	return hist, box_info, mag, frame_OF, elap


def optical_flow_mask(frame, frame_prev, boxes, confidences, classIDs, masks, LABELS, COLORS, oneobject=True, nb_bins=8, ratio=2):
	'''
	Inputs:
	frame: current frame
	frame_prev: previous frame
	boxes: boxes of the detected objects
	confidences: confidences of detected objects 
	classIDs: indices of the classes objects belong to; use together with LABELS to get the corresponding labels 
	masks: masks of the object
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
	if len(boxes)==0:
		print('WARNING: No object detected! Adding NaN values to features.')
		hist = np.full((1, nb_bins), np.nan)
		box_info = np.full((1, 4), np.nan)
		mag = np.full((1, 5), np.nan)
	else:
		if oneobject:
			idxs = [np.argmax(np.array(confidences))]
			# TODO: save features for multiobjects
			for i in idxs:
				# Attention: mask need to be modified as well
				xs, ys, xl, yl, mask = expand_box(boxes[i], masks[i], frame.shape[1], frame.shape[0], ratio=ratio)
				box_info = np.expand_dims(np.array([(xs+xl)/2, (ys+yl)/2, int(xl-xs), int(yl-ys)]), axis=0)
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
					magnitude[mask].mean(), # avg magnitude
					flow_horizontal[np.logical_and(flow_horizontal>=0, mask)].mean(),  # up
					flow_horizontal[np.logical_and(flow_horizontal<=0, mask)].mean(),  # down
					flow_vertical[np.logical_and(flow_vertical<=0, mask)].mean(),  # left
					flow_vertical[np.logical_and(flow_vertical>=0, mask)].mean()  # right
				])
				mag = np.asarray(mag)
				hist = HOOF(magnitude, angle, nb_bins, mask=mask, fuzzy=False, normalize=False)
				hsv = np.zeros_like(frame[ys:yl, xs:xl, :])
				hsv[..., 0] = angle*180/np.pi/2
				hsv[..., 1] = 255
				hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
				bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
				frame_OF[ys:yl, xs:xl, :] = bgr
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],	confidences[i])
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv.putText(frame_OF, text, (xs, ys - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
				blended = ((0.2 * np.array(color)) + (0.8 * bgr[mask])).astype("uint8")
				frame_OF[ys:yl, xs:xl, :][mask] = blended
				cv.imshow("object detection + optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return hist, box_info, mag, frame_OF, elap