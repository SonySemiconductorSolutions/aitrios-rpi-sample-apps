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

from modlib.apps import Annotator, BYTETracker, ObjectCounter
from modlib.devices import AiCamera
from modlib.models.zoo import SSDMobileNetV2FPNLite320x320


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

#-----Camera and AI setup-----
device = AiCamera()
model = SSDMobileNetV2FPNLite320x320()
device.deploy(model)

tracker = BYTETracker(BYTETrackerArgs())
people_counter = ObjectCounter()
annotator = Annotator(thickness=1, text_thickness=1, text_scale=0.4)

with device as stream:
    for frame in stream:
        #-----Detection Filtering-----
        detections = frame.detections[frame.detections.confidence > 0.55]
        detections = detections[detections.class_id == 0]  # Person
        #-----Tracker-----
        detections = tracker.update(frame, detections)

        people_counter.update(detections)
        #-----Display Annotations-----
        annotator.set_label(
            image=frame.image,
            x=430,
            y=30,
            color=(200, 200, 200),
            label="Total people detected " + str(people_counter.get(0)),
        )

        labels = [f"#{t} {model.labels[c]}: {s:0.2f}" for _, s, c, t in detections]
        annotator.annotate_boxes(frame=frame, detections=detections, labels=labels)

        frame.display()