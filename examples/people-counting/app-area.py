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

import json
import argparse
import cv2

from modlib.apps.annotate import ColorPalette, Annotator, Color
from modlib.apps.area import Area
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker


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
            
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        help="Json file containing bboxes of areas",
    )
    return parser.parse_args()
    
def start_area_count_demo():
    #-----Camera and AI setup-----
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera()
    device.deploy(model)

    json_areas = json_regions_extraction(args.json_file)
    areas = []
    for area in json_areas: 
        areas.append(Area(area["points"]))

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    with device as stream:
        for frame in stream:
            #-----Camera and AI setup-----
            detections = frame.detections[frame.detections.confidence > 0.5]
            detections = detections[detections.class_id == 0]
            #-----Tracker-----
            detections = tracker.update(frame, detections)
            #-----Display Annotations-----
            labels = [f"{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]

            frame.image = annotator.annotate_boxes(
                frame=frame,
                detections=detections,
                labels=labels,
                color=Color(0, 255, 255),
                alpha=0.2,
            )
            for ID, area in enumerate(areas):
                #-----Area-----
                d = detections[area.contains(detections)]
                #-----Visualize Detections-----
                frame.image = annotator.annotate_area(
                    frame=frame, area=area, color=(0, 255, 255), alpha = 0.2,
                )
                text_labels = [
                    "In Area: " + str(sum(1 for x in d if x)), #Get Number of people in each Area
                    "Area ID: " + str(ID + 1),
                ]

                for index, label in enumerate(text_labels):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_width, text_height = cv2.getTextSize(
                        text=label,
                        fontFace=font,
                        fontScale=0.5,
                        thickness=1,
                    )[0]
                    annotator.set_label(
                        image=frame.image,
                        x=int(((area.points[0][0] +  area.points[1][0]) / 2) * frame.width) - int(text_width/2),
                        y=int(((area.points[0][1] +  area.points[2][1]) / 2)* frame.height + ((index) * 25)) - int(2 * text_height),
                        color=(0, 255, 255),
                        label=label,
                    )
            frame.display()


if __name__ == "__main__":
    start_area_count_demo()
