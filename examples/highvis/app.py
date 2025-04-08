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

import numpy as np
import argparse
import os
import cv2

from modlib.apps.tracker.byte_tracker import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.devices import AiCamera
from typing import List, Optional

from modlib.models.model import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.devices.frame import Frame, IMAGE_TYPE
from modlib.models.results import Detections
from modlib.models.post_processors import pp_od_bscn
from modlib.apps.matcher import Matcher
from modlib.apps.object_counter import ObjectCounter


class Custom_Nanodet(Model):
    def __init__(self, custom_model_file, labels):
        super().__init__(
            model_file=custom_model_file,
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)


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
            frame.image # result is stored here
        )

        cv2.rectangle(
            img=frame.image,
            pt1=(x1, y1),
            pt2=(x2, y2),
            color=colour,
            thickness=2
        )

        if skip_label:
            continue
        label = f"{detections.class_id}" if (labels is None or len(detections) != len(labels)) else labels[i]

        annotator.set_label(image=frame.image, x=x1, y=y1, color=colour, label=label)

    return frame.image


class BYTETrackerArgs:
    track_thresh: float = 0.3
    track_buffer: int = 300
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path of the model")
    return parser.parse_args()


def start_highvis_demo():
    args = get_args()

    # ASSETS_DIR = f"{os.path.dirname(os.path.abspath(args.model))}"
    model = Custom_Nanodet(
        custom_model_file=args.model,
        labels=f"{os.path.dirname(os.path.abspath(args.model))}/labels.txt",
    )

    device = AiCamera()
    device.deploy(model)
    annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

    # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
    tracker = BYTETracker(BYTETrackerArgs())
    matcher = Matcher()
    total_counter = ObjectCounter()
    matched_counter = ObjectCounter()

    with device as stream:
        for frame in stream:
            detections = frame.detections[frame.detections.confidence > 0.4]
            detections = tracker.update(frame, detections)

            total_counter.update(detections)

            # Split your detections by Classes you wish to detect
            # person detections
            person_detections = detections[detections.class_id == 1]
            # safety-equipment detections
            vest_detections = detections[detections.class_id == 7]

            # Match  People with any class detections like a vest
            matched_people = person_detections[matcher.match(person_detections, vest_detections)]
            matched_counter.update(matched_people)
            m_labels = [f"{t}: Compliant     " for _, s, c, t in matched_people]
            p_labels = [f"{t}: Non Compliant" for _, s, c, t in person_detections]

            text_labels = [
                "Total people detected " + str(total_counter.get(1)),
                "Total people missing vest: " + str(total_counter.get(1) - matched_counter.get(1)),
            ]
            for index, label in enumerate(text_labels):
                annotator.set_label(
                    image=frame.image,
                    x=int(430),
                    y=int(30 + ((index) * 23)),
                    color=(200, 200, 200),
                    label=label,
                )

            frame.image = custom_annotate_boxes(
                frame=frame,
                detections=person_detections,
                annotator=annotator,
                labels=p_labels,
                colour=[0, 0, 255],
            )
            frame.image = custom_annotate_boxes(
                frame=frame,
                detections=matched_people,
                annotator=annotator,
                labels=m_labels,
                colour=[0, 255, 0],
            )

            frame.display()


if __name__ == "__main__":
    start_highvis_demo()
    exit()
