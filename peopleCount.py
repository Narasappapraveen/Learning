#	--output output/webcam_output.avi

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

import pdb

from detectionModule import DetectionModule

class PeopleCount():

	def __init__(self, args):
		self.args = args
		# initialize the total number of frames processed thus far, along
		# with the total number of objects that have moved either up or down
		self.totalFrames = 0
		self.totalDown = 0
		self.totalUp = 0
		self.trackableObjects = {}

	def run(self):
		# if a video path was not supplied, grab a reference to the webcam
		if not self.args.get("input", False):
			print("[INFO] starting video stream...")
			vs = VideoStream(src=0).start()
			time.sleep(1.0)
			fps = self.args["fps"]

		# otherwise, grab a reference to the video file
		else:
			print("[INFO] opening video file...")
			vs = cv2.VideoCapture(self.args["input"])
			fps = vs.get(cv2.CAP_PROP_FPS)

		# initialize the video writer (we'll instantiate later if need be)
		writer = None

		# initialize the frame dimensions (we'll set them as soon as we read
		# the first frame from the video)
		W = None
		H = None

		# instantiate our centroid tracker, then initialize a list to store
		# each of our dlib correlation trackers, followed by a dictionary to
		# map each unique object ID to a TrackableObject
		ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
		trackers = []

		object_detect = DetectionModule('person', self.args["confidence"])
		object_detect.load_model()
		
		with open('people_counter.csv', 'w') as f:
			csv_writer = csv.writer(f)		
			fields=['Incoming','Outgoing']
			csv_writer.writerow(fields)
		f.close()
		
		# loop over frames from the video stream
		while True:
			# grab the next frame and handle if we are reading from either
			# VideoCapture or VideoStream
			frame = vs.read()
			frame = frame[1] if self.args.get("input", False) else frame

			# if we are viewing a video and we did not grab a frame then we
			# have reached the end of the video
			if self.args["input"] is not None and frame is None:
				break

			# resize the frame to have a maximum width of 500 pixels (the
			# less data we have, the faster we can process it), then convert
			# the frame from BGR to RGB for dlib
			frame = imutils.resize(frame, width=500)
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			# if the frame dimensions are empty, set them
			if W is None or H is None:
				(H, W) = frame.shape[:2]

			# if we are supposed to be writing a video to disk, initialize
			# the writer
			if self.args["output"] is not None and writer is None:
				fourcc = cv2.VideoWriter_fourcc(*"MJPG")
				writer = cv2.VideoWriter(self.args["output"], fourcc, 30,
					(W, H), True)

			# initialize the current status along with our list of bounding
			# box rectangles returned by either (1) our object detector or
			# (2) the correlation trackers
			rects = []

			# check to see if we should run a more computationally expensive
			# object detection method to aid our tracker
			if self.totalFrames % self.args["skip_frames"] == 0:
				# set the status and initialize our new set of object trackers
				
				trackers = []

				detections = object_detect.inference(frame)
				#pdb.set_trace()
				print(self.totalFrames)		
				# loop over the detections
				for i in np.arange(0, len(detections)):

					box = detections[i]
					(startX, startY, endX, endY) = box.astype("int")
					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

			# otherwise, we should utilize our object *trackers* rather than
			# object *detectors* to obtain a higher frame processing throughput
			else:
				self.get_tracked_boxes(trackers, rects, rgb)

			self.assosciate_tracked_objects(rects, ct, frame, H, W, writer)
			for rect in rects:
				(startX, startY, endX, endY) = rect
				cv2.rectangle(frame,(startX, startY),(endX, endY) ,(0, 0, 255), 2)
			# show the output frame
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
				
			if self.totalFrames % fps == 0:
				with open('people_counter.csv', 'a') as f:
					csv_writer = csv.writer(f)	
					csv_writer.writerow([self.totalDown, self.totalUp])
				
				f.close()


		# check to see if we need to release the video writer pointer
		if writer is not None:
			writer.release()

		# if we are not using a video file, stop the camera video stream
		if not self.args.get("input", False):
			vs.stop()

		# otherwise, release the video file pointer
		else:
			vs.release()

		# close any open windows
		cv2.destroyAllWindows()

	def get_tracked_boxes(self, trackers, rects, rgb):
		# loop over the trackers
		for tracker in trackers:
			# set the status of our system to be 'tracking' rather
			# than 'waiting' or 'detecting'
			status = "Tracking"

			# update the tracker and grab the updated position
			tracker.update(rgb)
			pos = tracker.get_position()
			# unpack the position object
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# add the bounding box coordinates to the rectangles list
			rects.append((startX, startY, endX, endY))

	def assosciate_tracked_objects(self, rects, ct, frame, H, W, writer):
		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = self.trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0:
						self.totalUp += 1
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0:
						self.totalDown += 1
						to.counted = True

			# store the trackable object in our dictionary
			self.trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		# construct a tuple of information we will be displaying on the
		# frame
		info = [
			("Exit", self.totalUp),
			("Entry", self.totalDown),
		]

		# loop over the info tuples and draw them on our frame
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (W - 170 , (i*20)+30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# increment the total number of frames processed thus far
		self.totalFrames += 1
		
