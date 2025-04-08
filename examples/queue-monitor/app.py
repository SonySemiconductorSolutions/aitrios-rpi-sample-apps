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
import cv2
import numpy as np
#--------------------MODLIB IMPORTS---------------------------
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.devices.frame import Frame, IMAGE_TYPE
from modlib.models.results import Detections
from modlib.apps.area import Area
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker
from typing import List, Tuple, Optional

# custom annotate boxes
def custom_annotate_boxes(
    frame: Frame,
    detections: Detections,
    colour: List[int],
    annotator: Annotator,
    labels: Optional[List[str]] = None,
    skip_label: bool = False,
) -> np.ndarray:
    if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
        detections.compensate_for_roi(frame.roi)

    h, w, _ = frame.image.shape
    for i in range(len(detections)):
        overlay = frame.image.copy()
        x1, y1, x2, y2 = detections.bbox[i]

        # Rescaling to frame size
        x1, y1, x2, y2 = (
            int(x1 * w),
            int(y1 * h),
            int(x2 * w),
            int(y2 * h),
        )

        cv2.rectangle(
            img=frame.image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=colour,
            thickness=-1,
        )

        cv2.addWeighted(
            overlay,
            0.8,
            frame.image,
            0.2, 
            0, 
            frame.image
        )

        cv2.rectangle(img=frame.image, pt1=(x1, y1), pt2=(x2, y2), color=colour, thickness=2 )

        if skip_label:
            continue
        label = f"{detections.class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]

        annotator.set_label(image=frame.image, x=x1, y=y1, color=colour, label=label)

    return frame.image

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
        cv2.addWeighted(
            overlay,
            0.8,
            frame.image,
            0.2, 
            0, 
            frame.image 
        )

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
        help="Json file containing bboxes of queues",
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


def start_queue_manager_demo():
    args = get_args()

    model = NanoDetPlus416x416()
    device = AiCamera()
    device.deploy(model)

    queue_area = json_regions_extraction(args.json_file)
    areas = []
    for queue in queue_area:  # Change points to enter in single Areas
        areas.append(Area(queue["points"]))

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    # class_ids = [2, 5, 7]  # car, bus, truck
    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.5]
            detections = detections[detections.class_id == 0]
            detections = tracker.update(frame, detections)
            labels = [f"{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]

            frame.image = custom_annotate_boxes(
                frame=frame,
                detections=detections,
                annotator=annotator,
                labels=labels,
                colour=[255, 255, 0]

            )
            for ID, area in enumerate(areas):
                d = detections[area.contains(detections)]
                frame.image = custom_annotate_area(
                    frame=frame, area=area, annotator=annotator,color=(0, 255, 255)
                )
                text_labels = [
                    "In Queue: " + str(sum(1 for x in d if x)),
                    "Queue ID: " + str(ID + 1),
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
                        y=int(((area.points[0][1] +  area.points[2][1]) / 2)* frame.height - ((index) * 23)) + int(2 * text_height),
                        color=(0, 255, 255),
                        label=label,
                    )
            frame.display()


if __name__ == "__main__":
    start_queue_manager_demo()
