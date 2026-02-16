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
import argparse
import numpy as np

# --------------------MODLIB IMPORTS---------------------------
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.devices import AiCamera
from modlib.devices.frame import ROI
from modlib.models.zoo import NanoDetPlus416x416
from modlib.apps.tracker.byte_tracker import BYTETracker
from configurator import CVMenuStateMachine, config


def parse_roi(roi_str: str) -> ROI:
    try:
        # Split the input and parse the float values
        values = list(map(float, roi_str.split(",")))
        if len(values) != 4:
            raise ValueError
        return ROI(*values)
    except ValueError:
        raise argparse.ArgumentTypeError("ROI must be in the format 'left,top,width,height' with 4 float values.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_input", type=parse_roi, default=None, help="Set input tensor ROI (%(default)s).")
    return parser.parse_args()


class BYTETrackerArgs:
    track_thresh: float = 0.30
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def start_parking_management_demo():
    # -----Camera and AI setup-----
    args = get_args()
    gs = CVMenuStateMachine()
    roi = config.get_roi()
    model = NanoDetPlus416x416()
    device = AiCamera()
    device.deploy(model)
    if args.roi_input is not None:
        roi_it = args.roi_input
    else:
        roi_it = roi

    device.set_input_tensor_cropping(tuple(roi_it))
    device.set_image_cropping(tuple(roi_it))

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)
    class_ids = [2, 5, 7]  # car, bus, truck
    with device as stream:
        for frame in stream:
            gs.tick(frame)
            occupied = 0
            # -----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.5]
            detections = detections[np.isin(detections.class_id, class_ids)]
            # -----Tracker-----
            detections = tracker.update(frame, detections)
            # -----Display Annotations-----
            labels = [f"{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
            frame.image = annotator.annotate_boxes(frame=frame, detections=detections, labels=labels, alpha=0.2)

            for ID, area in enumerate(config.get_areas()):
                # -----Filter Area Detections-----
                d = detections[area.contains(detections)]
                # -----Display Areas-----
                if d:
                    frame.image = annotator.annotate_area(frame=frame, area=area, color=(0, 0, 255), alpha=0.2)
                    occupied += 1
                else:
                    frame.image = annotator.annotate_area(frame=frame, area=area, color=(0, 255, 0), alpha=0.2)

            text_labels = [
                "Occupied: " + str(occupied),
                "Free Spaces: " + str(len(config.get_areas()) - occupied),
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
