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

import numpy as np
import argparse
import os
import json

from modlib.apps.tracker.byte_tracker import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator, Color
from modlib.devices import AiCamera
from typing import List

from modlib.models.model import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.results import Detections
from modlib.models.post_processors import pp_od_bscn
from modlib.apps.matcher import Matcher
from modlib.apps.object_counter import ObjectCounter


class Custom_Nanodet(Model):
    def __init__(self, custom_model_file, labels):
        super().__init__(
            model_file=custom_model_file,
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.BGR,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(labels, dtype=str, delimiter="\n")

    def pre_process(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Pre-processing not implemented for this model.")

    def post_process(self, output_tensors: List[np.ndarray]) -> Detections:
        return pp_od_bscn(output_tensors)

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
    #-----Camera and AI setup-----
    args = get_args()

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
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.4]
            #-----Tracker-----
            detections = tracker.update(frame, detections)
            # Split your detections by Classes you wish to detect
            person_detections = detections[detections.class_id == 1]
            vest_detections = detections[detections.class_id == 7]
            total_counter.update(detections)
            #-----Matcher-----
            matched_people = person_detections[matcher.match(person_detections, vest_detections)]
            matched_counter.update(matched_people)
            #-----Display Annotations-----
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

            frame.image = annotator.annotate_boxes(
                frame=frame,
                detections=person_detections,
                labels=p_labels,
                color=Color(255, 0, 0),
                alpha = 0.2,
            )
            frame.image = annotator.annotate_boxes(
                frame=frame,
                detections=matched_people,
                labels=m_labels,
                color=Color(0, 255, 0),
                alpha = 0.2,
            )
            
            frame.display()
            
            #-----Output to json-----
            #Detections - bbox, class_id, scores, tracker_id
            #Matcher - bbox, avg_overlap, class_id, tracker_id, overlapped, uptime, missing_tracker_counter
            
            objects = {}
            roi = frame.roi #[0.1252465, 0.0, 0.7495069, 1.0]
            for bbox, overlap, class_id, tracker_id, overlapped, up_time, missing_counter in matcher:
                if tracker_id is not None:
                    tracker_id = int(tracker_id)
                    h, w, _ = frame.image.shape
                    objects[tracker_id] = {
                    "vest": overlapped, 
                    "overlap_value": float(overlap), 
                    "bbox": {"X": int((roi[0] + bbox[0] * roi[2])*w), "Y": int((roi[1] + bbox[1] * roi[3])*h), "x": int((roi[0] + bbox[2] * roi[2])*w), "y": int((roi[1] + bbox[3] * roi[3])*w)},
                    "uptime": int(up_time),
                    }
                    if up_time < 30:
                        objects[tracker_id]["status"] = "STABILIZING"
                    elif up_time == 30 and missing_counter < 2:
                        objects[tracker_id]["status"] = "NEW"
                    else:
                        objects[tracker_id]["status"] = "TRACKING"
            for deleted_object in matcher.deleted_ids:
                objects[int(deleted_object)] = {"status": "LOST"}
            print(objects)
            with open('data.json', "a", encoding="utf-8") as f:
                json.dump(objects, f)
                f.write('\n')
            

if __name__ == "__main__":
    start_highvis_demo()
    exit()
