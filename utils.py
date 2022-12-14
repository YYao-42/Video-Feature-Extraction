import numpy as np
import cv2 as cv
import time
import copy


def draw_labels_and_boxes(img, boxes, confidences, classids, idxs, colors, labels):
	# If there are any detections
	if len(idxs) > 0:
		for i in idxs.flatten():
			# Get the bounding box coordinates
			x, y = boxes[i][0], boxes[i][1]
			w, h = boxes[i][2], boxes[i][3]
			
			# Get the unique color for this class
			color = [int(c) for c in colors[classids[i]]]

			# Draw the bounding box rectangle and label on the image
			cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
			# text = "{}: {:4f}".format(labels[classids[i]], confidences[i])
			# cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return img


def object_detection_yolo(frame, net, ln, W, H, args):
	'''
	Inputs:
	frame: current frame
	net: yolo net for object detection
	ln: layer names of layers to be retrieved
	W: width of the frame
	H: height of the frame
	args: pre-defined parameters
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
			if confidence > args["confidence"]:
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


def optical_flow_FB(idxs, boxes, frame, frame_prev):
	start = time.time()
	cv.imshow("input", frame)
	frame_OF = copy.deepcopy(frame)
	frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
	frame_prev_grey = cv.cvtColor(frame_prev, cv.COLOR_BGR2GRAY)
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		patch = frame_grey[y:y+h, x:x+w]
		patch_prev = frame_prev_grey[y:y+h, x:x+w]
		flow_patch = cv.calcOpticalFlowFarneback(patch_prev, patch, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		# flow_horizontal = flow_patch[..., 0]
		# flow_vertical = flow_patch[..., 1]
		# Computes the magnitude and angle of the 2D vectors
		magnitude, angle = cv.cartToPolar(flow_patch[..., 0], flow_patch[..., 1])
		hsv = np.zeros_like(frame[y:y+h, x:x+w, :])
		hsv[..., 0] = angle*180/np.pi/2
		hsv[..., 1] = 255
		hsv[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
		bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
		frame_OF[y:y+h, x:x+w, :] = bgr
		cv.imshow("dense optical flow", frame_OF)
	end = time.time()
	elap = end - start
	return frame_OF, elap