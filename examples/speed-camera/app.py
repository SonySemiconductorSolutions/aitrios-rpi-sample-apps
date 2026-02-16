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
from modlib.apps.calculate import SpeedCalculator
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--json-file", type=str, required=True, default= "object_sizes.json", help="Json file containing bboxes of queues")
	return parser.parse_args()

def annotate_speed_area(
        frame, area, color
    ) -> np.ndarray:

        h, w, _ = frame.image.shape
        resized_points = np.empty(area.region.shape, dtype=np.int32)
        resized_points[:, 0] = (area.region[:, 0] * w).astype(np.int32)
        resized_points[:, 1] = (area.region[:, 1] * h).astype(np.int32)
        resized_points = resized_points.reshape((-1, 1, 2))

        # Draw the area on the image
        cv2.polylines(frame.image, [resized_points], isClosed=True, color=color, thickness=2)

        return frame.image

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

def start_speed_demo():
	#-----Camera and AI setup-----
	args = get_args()
	model = NanoDetPlus416x416()
	device = AiCamera(frame_rate=17)
	device.deploy(model)
	
	calibration = json_regions_extraction(args.json_file)
	speed_areas = []
	dpp = []
	for area in calibration: # Change points to enter in single Areas
		t_area = Area(area["points"])
		speed_areas.append(SpeedCalculator(t_area.points))
		dpp.append(area["dpp"])
	
	# Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
	tracker = BYTETracker(BYTETrackerArgs())
	annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)
	class_ids =[2,3,5,7] # car, motorcycle, bus, truck detections
	with device as stream:
		for frame in stream:
			#-----Detection Filtering-----
			detections = frame.detections[frame.detections.confidence > 0.4]
			detections = detections[np.isin(detections.class_id, class_ids)]
			#-----Tracker-----
			detections = tracker.update(frame, detections)
			#-----Speed Calculate-----
			for ID, speed in enumerate(speed_areas):
				frame.image = annotate_speed_area(frame=frame, area=speed, color=(0,0,255))
				speed.calculate(frame, detections)

			labels = []		
			for t in detections.tracker_id:
				average_speed = []
				stationary = False
				for ID, speed in enumerate(speed_areas):
					if t == -1:
						continue
					if t in speed.tracked_id:
						average_speed.append((speed.get_speed(t, average=True))*dpp[ID]*3.6)
					else:
						if speed.stationary[t]:
							stationary = True
				#-----Display Annotations-----		
				if len(average_speed) > 0:
					labels.append(f"speed: {(sum(average_speed)/len(average_speed)):0.2f}kph") #Average speed through box
				else:
					if stationary:
						labels.append("speed: 0.00kph") #First calculated speed
					else:
						labels.append("calculating")	

			
			frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)
			
			frame.display()
			
if __name__ == "__main__":
	start_speed_demo()
