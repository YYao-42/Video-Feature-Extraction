import numpy as np
import cv2 as cv
import time
import copy


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


def optical_flow_FB(frame, frame_prev, boxes, confidences, classIDs, idxs, LABELS, COLORS, oneobject=True):
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
	frame_OF: modified current frame 
	elap: processing time
	'''
	start = time.time()
	cv.imshow("input", frame)
	frame_OF = copy.deepcopy(frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	idxs = idxs.flatten()
	if len(idxs)==0:
		print('WARNING: No object detected! Adding NaN values to the feature array.')
		# TODO:
	else:
		if oneobject:
			idx_maxconfi = np.argmax(np.array(confidences)[idxs])
			idxs = [idxs[idx_maxconfi]]
		for i in idxs:
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
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
			# flow_horizontal = flow_patch[..., 0]
			# flow_vertical = flow_patch[..., 1]
			# Computes the magnitude and angle of the 2D vectors
			magnitude, angle = cv.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])
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
	return frame_OF, elap