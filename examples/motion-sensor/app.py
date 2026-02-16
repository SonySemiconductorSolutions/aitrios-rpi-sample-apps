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

# ----------------------IMPORTS--------------------------
import json
import argparse
import cv2
from pathlib import Path
#--------------------MODLIB IMPORTS---------------------------
from modlib.apps.annotate import ColorPalette, Annotator, Color

from modlib.models.results import Detections
from modlib.apps.area import Area
from modlib.devices.frame import Frame
from modlib.apps.matcher import Matcher
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker

from modlib.apps.motion import Motion

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file",type=str,required=True,default=None,help="Json file containing bboxes of check",)
    parser.add_argument("--area",action="store_true",help="Bool to toogle visualize of the Area or not",)
    return parser.parse_args()

def check(detections, matcher ,motion):
    _check = False
    if len(detections) <= 0 and motion.constant_motion == motion.motion_threshold: #add user flag here to turn on 
        _check = True
    else:
        for obj in matcher:
            if obj[4] and obj[5] > 10 : #if matched is true and uptime for 10 frames
                if obj[3] not in motion.image_IDs:
                    motion.image_IDs.append(obj[3])
                    _check = True
                elif obj[3] in motion.image_IDs and obj[6] > 10: #If ID in stored and missing for 10 frames
                    motion.image_IDs = [i for i in motion.image_IDs if i != obj[3]]
            elif obj[3] in motion.image_IDs and not obj[4]: #If ID in stored and not moving
                motion.image_IDs = [i for i in motion.image_IDs if i != obj[3]]
    return _check
                
def save_frame(frame: Frame, detections: Detections, matcher: Matcher, motion: Motion, path: str = "./images"):
    directory = Path(path)
    directory.mkdir(parents = True, exist_ok=True)
    if check(detections,matcher,motion):
        output_path = directory / f"{frame.timestamp}.jpg"
        cv2.imwrite(str(output_path),frame.image)
        print(f'Motion detected at {str(frame.timestamp)}: image captured')
    return
        
class BYTETrackerArgs:
    track_thresh: float = 0.30
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
        
def json_regions_extraction(json_filename):
    """
    Extract queue regions from json file.
    """
    with open(json_filename, "r") as json_file:
        area_pts = json.load(json_file)
        if len(area_pts) > 0:
            return area_pts
        else:
            raise Exception("Please ensure there are areas to check")

def start_motion_manager_demo():
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera(frame_rate=15)
    device.deploy(model)

    motion_area = json_regions_extraction(args.json_file)
    areas = []
    for area in motion_area:  # Change points to enter in single Areas
        areas.append(Area(area["points"]))

    motion = Motion()
    matcher = Matcher(max_missing_overlap=10, max_missing_tracker =10)

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    with device as stream:
        for frame in stream:
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.5]
            detections = detections[detections.class_id == 0]
            #-----Tracker-----
            detections = tracker.update(frame, detections)
            #-----Motion-----
            motion_bboxes = motion.detect(frame)
            #-----Visualize All Detections-----
            labels = [f"{t} {model.labels[c]}" for _, s, c, t in detections]
            frame.image = annotator.annotate_boxes(frame=frame,detections=detections,labels=labels,color=Color(255, 0, 0))
            
            #-----Check Areas-----
            for area in areas: #Check if detections is in marked area
                area_detections = detections[area.contains(detections)]
                area_motion = motion_bboxes[area.contains(motion_bboxes)]
                labels = [f"{t}: in area"  for _, _, _, t in area_detections]
                frame.image = annotator.annotate_boxes(frame=frame,detections=area_detections,labels=labels,color=Color(0, 255, 255))
                if args.area:
                    frame.image = annotator.annotate_area(frame=frame, area=area, color=(0, 255, 255), alpha = 0.2,)
                    
            #-----Visualize Motion-----
            frame.image = annotator.annotate_boxes(frame,motion_bboxes,color=Color(0, 255, 255),skip_label=True)
            #-----Matcher-----
            motion_detections = detections[matcher.match(area_detections,area_motion)]
                       
            #-----Visualize Moving Detections-----
            labels = [f"{t}: moving"  for _, _, _, t in motion_detections ]
            frame.image = annotator.annotate_boxes(frame=frame,detections=motion_detections,labels=labels,color=Color(0, 255, 0))
            
            #-----Save Video-----
            save_frame(frame, detections,matcher, motion) #Save frames
            frame.display()


if __name__ == "__main__":
    start_motion_manager_demo()
