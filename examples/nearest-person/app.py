#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import json
import argparse
import numpy as np

from modlib.apps import Annotator, Area, ColorPalette
from modlib.apps.annotate import Color
from modlib.apps.calculate import calculate_distance_matrix
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--json-file", type=str, required=True, default= "object_sizes.json", help="Json file containing bboxes of queues")
	return parser.parse_args()

class BYTETrackerArgs:
	track_thresh: float = 0.30
	track_buffer: int = 30
	match_thresh: float = 0.8
	aspect_ratio_thresh: float = 3.0
	min_box_area: float = 1.0
	mot20: bool = False

def json_regions_extraction(json_filename):
	with open(json_filename, "r") as json_file: 
		calibration = json.load(json_file)
		if len(calibration) > 0:
			print(calibration)
			return calibration
		else:
			raise Exception("Please ensure there are areas to check")

def nearest_neighbours(detections, frame, areas, dpp, annotator):
	labels = []
	
	xc, yc = detections.center_points
	xc, yc = xc*frame.width, yc*frame.height
	dist_matrix = calculate_distance_matrix(xc, yc)
	if len(dist_matrix) > 0:
		np.fill_diagonal(dist_matrix, np.inf)
		closest_indices = np.argmin(dist_matrix,axis = 1)
		
		for i, closest_index in enumerate(closest_indices):
			pixel_distance1,pixel_distance2 = 0 ,0
			dist = dist_matrix[i, closest_index]
			p1 = (int(xc[i]), int(yc[i]))
			p2 = (int(xc[closest_index]), int(yc[closest_index]))
			
			for ID, area in enumerate(areas):
				if cv2.pointPolygonTest(area.points, (p1[0]/frame.width, p1[1]/frame.height), False) >= 0:
					pixel_distance1 = dpp[ID]
				if cv2.pointPolygonTest(area.points, (p2[0]/frame.width, p2[1]/frame.height), False) >= 0:
					pixel_distance2 = dpp[ID]			
			average_pd = (pixel_distance1 + pixel_distance2)/2
			real_dist = round((dist * average_pd),2)
			labels.append(f"Closest person: {str(real_dist)}m")
			if dist* average_pd > 2:
				cv2.arrowedLine(frame.image, p1, p2, [128,255,0], 2)
			else:
				cv2.arrowedLine(frame.image, p1, p2, [0,0,255], 2)
	frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels, color=Color(0,255,255))
	
	return frame.image
def start_distance_demo():
	#-----Camera and AI setup-----
	args = get_args()
	model = NanoDetPlus416x416()
	device = AiCamera()
	device.deploy(model)
	
	depth = json_regions_extraction(args.json_file)
	areas = []
	dpp = []
	for area in depth:#Change points to enter in single Areas
		areas.append(Area(area["points"]))
		dpp.append(area["dpp"])
	
	# Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
	tracker = BYTETracker(BYTETrackerArgs())
	annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)
	with device as stream:
		for frame in stream:
			#-----Detection Filtering-----
			detections = frame.detections[frame.detections.confidence > 0.40]
			detections = detections[detections.class_id == 0]
			#-----Tracker-----
			detections = tracker.update(frame, detections)
			#-----Application Logic and Display-----
			frame.image = nearest_neighbours(detections, frame,areas,dpp,annotator)
			frame.display()
			
if __name__ == "__main__":
	start_distance_demo()
