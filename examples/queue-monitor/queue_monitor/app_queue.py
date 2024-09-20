"""
  Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import cv2

import json
import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import  Tuple

from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, MappedArray, CompletedRequest

from queue_monitor.matcher.object_event import ObjectEventLogic, set_settings as object_event_settings
from queue_monitor.matcher.example.utils import *
from queue_monitor.matcher.tracker_helper import *
from queue_monitor.tracker import BYTETracker

last_boxes = None
last_scores = None
last_keypoints = None

class ToTrack:
    output_results: np.ndarray
    img_info: Tuple[int, int]
    img_size: Tuple[int, int]

class QueueExample:
	def __init__(self):
		self.args = self.get_args()
		self.queue_regions_extraction()
		self.queue_counter = []
		# Delcare starting count for each queue
		for i in self.queue_pts: 
			self.queue_counter.append(0)
			
		self.BOX_MIN_CONFIDENCE = self.args.box_min_confidence
		self.IOU_THRESHOLD = self.args.iou_threshold
		self.MAX_OUT_DETS = self.args.max_out_dets
		
		self.settings_dict = self.parse_settings(settings("settings.json"))
		signal.signal(signal.SIGINT, shutdown_handler)
		
		# This must be called before instantiation of Picamera2
		self.imx500 = IMX500(self.args.model)
		self.tracker = BYTETracker(BYTETrackerArgs(self.settings_dict))
		
		self.picam2 = Picamera2()
		config = self.picam2.create_preview_configuration(controls={'FrameRate': self.args.fps}, buffer_count=28)
		self.picam2.start(config, show_preview=True)
		self.imx500.set_auto_aspect_ratio()
		self.picam2.pre_callback = self.run
		
	def queue_regions_extraction(self):
		"""
		Extract queue regions from json file.
		"""
		with open(self.args.json_file, "r") as json_file:
			queue_pts = json.load(json_file)
			if len(queue_pts) > 0:
				self.queue_pts = queue_pts
			else:
				raise Exception("Please ensure there are queues to check")
			return queue_pts

	def get_args(self):
		parser = argparse.ArgumentParser()
		parser.add_argument("--model", type=str, required=True, help="Path of the model")
		parser.add_argument("--fps", type=int, default=30, help="Frames per second")
		parser.add_argument("--json-file", type=str, required=True, default= None, help="Json file containing bboxes of queues")
		parser.add_argument("--box-min-confidence", type=float,
							default=0.3, help="Confidence threshold for bounding box predictions")
		parser.add_argument("--iou-threshold", type=float, default=0.7,
							help="IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS)")
		parser.add_argument("--max-out-dets", type=int, default=300,
							help="Maximum number of output detections to keep after NMS")
		return parser.parse_args()
		
	def run(self,request: CompletedRequest):
		boxes, scores, classes = self.ai_output_tensor_parse(request.get_metadata())
		self.ai_draw_queue_mgt(request, boxes, scores,classes)
		
	def format_track(self,boxes,scores):
		"""Format and get the tracker IDs for the detected objects"""
		out = ToTrack()
		out.img_info = [640,640]
		out.img_size = [640,640]
		out.output_results = np.hstack((np.array(boxes), scores[:,np.newaxis]))
		
		tracks = self.tracker.update(out)
		tracker_id =  match_detections_with_tracks(boxes, tracks)
		return tracker_id
		
	def ai_output_tensor_parse(self,metadata: dict):
		"""Parse the output tensor into a number of detected objects, scaled to the ISP out."""
		global last_boxes, last_scores, last_classes
		np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
		if np_outputs is not None:
			last_boxes, last_scores, last_classes = [], [], []
			boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
			for box, score, category in zip(boxes, scores, classes):
				if score > self.BOX_MIN_CONFIDENCE:
					last_boxes.append(box)
					last_scores.append(score)
					last_classes.append(category)
		return np.array(last_boxes), np.array(last_scores), last_classes
	

	def ai_draw_queue_mgt(self,request: CompletedRequest, boxes, scores,classes, stream='main'):
		"""Draw the detections for this request onto the ISP output."""
		
		with MappedArray(request, stream) as m:   
			results =[]
			b = self.imx500.get_roi_scaled(request)
			for i,queue in enumerate(self.queue_pts):
				cp = m.array
				pts = np.array(queue["points"]).reshape(-1,1,2)
				cv2.rectangle(
					cp,
					(queue["points"][0][0] , queue["points"][0][1] -43),
					(queue["points"][0][0] + 95, queue["points"][0][1] -5),
					(255,255,255),
					-1,
				)
				
				cp = cv2.addWeighted(cp,0.4,m.array,1-0.4,0)
				cv2.putText(m.array, "Queue ID: "+str(i+1), (queue["points"][0][0] , queue["points"][0][1] -30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (0,0,0), 1)
				cv2.putText(m.array, "In queue: "+str(self.queue_counter[i]), (queue["points"][0][0] , queue["points"][0][1] -10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.65, (0,0,0), 1)
				cv2.polylines(m.array, [pts], True , (255, 0, 0),2)
				self.queue_counter[i] = 0
			metadata = request.get_metadata()
			for i,queue in enumerate(self.queue_pts): #Loop for number of queues 
				queue_reshape = np.array(queue["points"], dtype=np.int32).reshape((-1, 1, 2))
				if boxes is not None:
					if len(boxes) > 0:
						tracker_id = self.format_track(boxes,scores)
						for j,box in enumerate(boxes): #Loop for number of detections
							if classes[j] == 0: #if class is person
								obj_scaled = self.imx500.convert_inference_coords(box, metadata, self.picam2)
								box = (obj_scaled.x, obj_scaled.y, obj_scaled.width, obj_scaled.height)
								x_center = int((box[0] + (box[0]+box[2])) / 2)
								y_center = int((box[1] + (box[1]+box[3])) / 2)
								
								cv2.putText(m.array, "ID: "+ str(tracker_id[j]), (box[0] + 5, box[1] + 15), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
								cv2.rectangle(m.array, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0, 0),2)
								#Calc to see if center person point is in a queue box 
								dist = cv2.pointPolygonTest(queue_reshape, (x_center, y_center), False) 
								if dist >= 0:
									self.queue_counter[i] = self.queue_counter[i] + 1
								cv2.rectangle(m.array, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0, 0),2)
							
	def parse_settings(self,settings_dict):
		"""Setup settings for model"""
		object_event_settings(settings_dict)
		model_fpk = os.path.abspath(settings_dict["IMX_CONFIG"]["NETWORK_FPK_PATH"])

		confidence_threshold = settings_dict["CONFIDENCE_THRESHOLD"]

		return settings_dict

	def get_point(self,keypoints, index,metadata):
		"""Scale the point to image coordinates for drawing on image"""
		y0, x0 = keypoints[index][1], keypoints[index][0]
		coords = y0, x0, y0 + 1, x0 + 1
		obj_scaled = self.imx500.convert_inference_coords(coords, metadata, self.picam2, stream="main")
		return obj_scaled.x, obj_scaled.y

def start_queue_manager_demo():
	cv2.startWindowThread()
	qm = QueueExample()
	qm.run
	while True:
		cv2.waitKey(30)

if __name__ == "__main__":
	start_queue_manager_demo()
	while True:
		cv2.waitKey(30)
	exit()

