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

from modlib.apps import Annotator, BYTETracker
from modlib.devices import AiCamera
from modlib.models.zoo import NanoDetPlus416x416
from modlib.devices.frame import Frame
from modlib.models import Detections
import numpy as np
import cv2
import random


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def crop_and_display(frame: Frame, image: np.ndarray, detection: Detections) -> np.ndarray:
    h, w, _ = frame.image.shape
    crop_h, crop_w = 160, 120
    crop_x1, crop_y1, crop_x2, crop_y2 = int(w - crop_w), int(0), int(w), int(crop_h)

    if detection.bbox.any():
        x1, y1, x2, y2 = int(detection.bbox[0][0] * w), int(detection.bbox[0][1] * h), int(detection.bbox[0][2] * w), int(detection.bbox[0][3] * h)
        cropped_region = annotator.crop(image, x1, y1, x2, y2)
        cropped_region = cv2.resize(cropped_region, (crop_w, crop_h))
        frame.image[crop_y1:crop_y2, crop_x1:crop_x2] = cropped_region

    return frame.image
#-----Camera and AI setup-----
device = AiCamera()
model = NanoDetPlus416x416()
device.deploy(model)

annotator = Annotator()
tracker = BYTETracker(BYTETrackerArgs())

tracker_id = -1

with device as stream:
    for frame in stream:
        #-----Detection Filtering-----
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person
        #-----Tracker-----
        detections = tracker.update(frame, detections)

        if tracker_id == -1 and (detections.tracker_id).size > 0:
            tracker_id = random.choice(detections.tracker_id)
        if tracker_id not in detections.tracker_id:
            tracker_id = -1
        target = detections[detections.tracker_id == tracker_id]
        #-----Display Annotations-----
        labels = [f"#{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
        copy_img = frame.image.copy()
        annotator.annotate_boxes(frame, detections, labels=labels, alpha= 0.2)
        crop_and_display(frame, copy_img, target)
    
        frame.display()