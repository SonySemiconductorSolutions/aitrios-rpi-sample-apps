"""
  Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

""" 
    This file defines a class used to create work objects for frame annotation tasks.
    These work objects are intended to be placed into a queue for processing.
    Each instance of the class represents a specific frame annotation job.
"""


class WorkQueueAnnotator:
    def __init__(self, detectors, annotators, settings):
        self.detectors = detectors
        self.annotators = annotators
        self.an_setting = settings["ANNOTATION"]
        self.labels = settings["IMX_CONFIG"]["LABELS"]

    def annotate(self, frame):
        if self.an_setting["BYTE_TRACKER"]:
            labels = []
            for bbox, confidence, class_id, tracker_id in self.detectors[0]:
                if class_id > len(self.labels) - 1:
                    print(f"class id: {class_id} labels: {self.labels}")
                labels += [f"#{tracker_id} {self.labels[class_id]} {confidence:0.2f}"]
            frame = self.annotators[0].annotate(scene=frame, detections=self.detectors[0], labels=labels)

        if self.an_setting["OBJECT_OVERLAY"]:
            iou_text = []
            for bbox, confidence, class_id, tracker_id, iou in self.detectors[1]:
                iou_text += [f"#{tracker_id} OL:{iou:0.2f}"]
            frame = self.annotators[1].annotate(scene=frame, detections=self.detectors[1], labels=iou_text)

        if self.an_setting["OBJECT_EVENT"]:
            logic_text = []
            for bbox, avg_overlap, class_id, tracker_id, overlap, uptime in self.detectors[2]:
                logic_text += [f"P:{tracker_id} {overlap}:{avg_overlap:0.1f} {uptime}"]
            frame = self.annotators[2].annotate(scene=frame, detections=self.detectors[2], labels=logic_text)
        return frame

    def annotate_frame_info(self, frame, fps, frame_number):
        return self.annotators[-1].annotate_frame_info(frame, fps, frame_number)
