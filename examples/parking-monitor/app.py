#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
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
import numpy as np
import cv2
#--------------------MODLIB IMPORTS---------------------------
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.apps.area import Area
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker
from typing import Tuple, Optional
from modlib.devices.frame import Frame


# custom annotate area
def custom_annotate_area(
        frame: Frame, area: Area, color: Tuple[int, int, int], annotator: Annotator, label: Optional[str] = None
    ) -> np.ndarray:
       
        overlay = frame.image.copy()
        h, w, _ = frame.image.shape
        resized_points = np.empty(area.points.shape, dtype=np.int32)
        resized_points[:, 0] = (area.points[:, 0] * w).astype(np.int32)
        resized_points[:, 1] = (area.points[:, 1] * h).astype(np.int32)
        resized_points = resized_points.reshape((-1, 1, 2))

        # Draw the area on the image
        cv2.fillPoly(frame.image, [resized_points], color=color)
        cv2.addWeighted(overlay, 0.8, frame.image, 0.2, 0, frame.image)
        cv2.polylines(frame.image, [resized_points], isClosed=True, color=color, thickness=2)

        # Label
        if label:
            annotator.set_label(
                image=frame.image, x=resized_points[0][0][0], y=resized_points[0][0][1], color=color, label=label
            )

        return frame.image

        
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-file",
        type=str,
        required=True,
        default=None,
        help="Json file containing bboxes of parking spaces",
    )
    return parser.parse_args()


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


def start_parking_management_demo():
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera()
    device.deploy(model)

    parking_spaces = json_regions_extraction(args.json_file)
    areas = []
    for space in parking_spaces:  # Change points to enter in single Areas
        areas.append(Area(space["points"]))

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    class_ids = [2, 5, 7]  # car, bus, truck
    with device as stream:
        for frame in stream:
            occupied = 0

            detections = frame.detections[frame.detections.confidence > 0.20]
            detections = detections[np.isin(detections.class_id, class_ids)]

            detections = tracker.update(frame, detections)
            labels = [f"{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
            frame.image = annotator.annotate_boxes(
                frame=frame, detections=detections, labels=labels
            )
            
            for ID, area in enumerate(areas):
                d = detections[area.contains(detections)]
                if d:
                    frame.image = custom_annotate_area(
                        frame=frame, area=area, color=(0, 0, 255), annotator=annotator
                    )
                    occupied += 1
                else:
                    frame.image = custom_annotate_area(
                        frame=frame, area=area, color=(0, 255, 0), annotator=annotator
                    )

            text_labels = [
                "Occupied: " + str(occupied),
                "Free Spaces: " + str(len(areas) - occupied),
            ]
            for index, label in enumerate(text_labels):
                annotator.set_label(
                    image=frame.image,
                    x=int(430),
                    y=int(30 + ((index) * 23)),
                    color=(200, 200, 200),
                    label=label,
                )
            frame.display()


if __name__ == "__main__":
    start_parking_management_demo()
