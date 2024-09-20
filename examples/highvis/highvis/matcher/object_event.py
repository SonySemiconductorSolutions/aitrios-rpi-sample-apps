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

from collections import deque
import numpy as np
from highvis.matcher.overlay_detector import OverlapDetector
from highvis.tracker import Detections
from typing import List, Tuple, Iterator, Union
import time

MAX_MISSING_OVERLAP = 60
MAX_MISSING_TRACKER = 30
MIN_OVERLAP_THRESHOLD = 0.5
HYSTERESIS = 0.4


"""
 The FIFOQueue class provides positional filtering, enabling smoother tracking of objects.
"""


class FIFOQueue:
    def __init__(self, n, initial_bbox):
        self.max_length = n
        self.bbox = np.tile(initial_bbox, (n, 1))
        self.current_index = 0

    def push(self, item):
        self.bbox[:-1] = self.bbox[1:]
        self.bbox[-1] = item
        self.current_index = (self.current_index + 1) % self.max_length

    def pop(self):
        popped_item = self.bbox[0]
        self.bbox[:-1] = self.bbox[1:]
        self.bbox[-1] = 0
        return popped_item

    def get_average(self):
        return np.mean(self.bbox, axis=0)

    def get_array(self):
        return self.bbox


"""
 The DetectedObject class encapsulates the state information of each detected object, 
 maintaining details such as position, state of overlap, and other relevant attributes.
"""


class DetectedObject:
    def __init__(self, bbox, confidence, class_id, tracker_id, overlap):
        self.bbox = bbox
        self.bboxes = FIFOQueue(8, bbox)
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = tracker_id
        self.overlap = overlap
        self.missing_tracker_counter = 0
        self.overlap_list = deque([overlap] * MAX_MISSING_OVERLAP)
        self.avg_overlap = 0
        self.overlapped = False
        self.uptime = 0

    def update(self, source):
        self.uptime += 1
        self.missing_tracker_counter = 0
        self.bboxes.push(source.bbox)
        self.bbox = self.bboxes.get_average()
        self.confidence = source.confidence

        if source.overlap > MIN_OVERLAP_THRESHOLD:
            self.overlap_list.clear()
            self.overlap_list.extend([source.overlap] * (MAX_MISSING_OVERLAP - 1))

        self.overlap_list.append(source.overlap)
        if len(self.overlap_list) > MAX_MISSING_OVERLAP:
            self.overlap_list.popleft()

    def missing(self):
        self.missing_tracker_counter += 1
        return self.missing_tracker_counter > MAX_MISSING_TRACKER

    def get(self):
        self.avg_overlap = 0
        if len(self.overlap_list):
            self.avg_overlap = sum(self.overlap_list) / len(self.overlap_list)

        if self.avg_overlap > MIN_OVERLAP_THRESHOLD + HYSTERESIS / 2:
            self.overlapped = True
        elif self.avg_overlap < MIN_OVERLAP_THRESHOLD - HYSTERESIS / 2:
            self.overlapped = False

        return (self.bbox, self.avg_overlap, self.class_id, self.tracker_id, self.overlapped, self.uptime)

    def __eq__(self, other):
        if isinstance(other, DetectedObject):
            return self.tracker_id == other.tracker_id
        return False

    def is_new(self):
        if self.uptime == 0:
            return True
        return False

    def __str__(self):
        return f"bbox {self.bbox}\nconfidence {self.confidence}\nclass_id {self.class_id}\ntracker_id {self.tracker_id}\noverlap {self.overlap}\n"


"""
 The ObjectEventLogic class is responsible for parsing new detections, matching them with the current state of objects, 
 and ensuring that each object's state is updated accordingly. This class plays a crucial role in maintaining the accuracy 
 and consistency of object tracking by synchronizing detected events with existing object data.
"""


class ObjectEventLogic(Detections):
    def __init__(self, detections: Detections):
        super().__init__(detections.frame_size[0], detections.frame_size[0])
        self.tracked_objects = deque()
        self.filtered_tracked_objects = deque()
        self.total = 0
        self.without = 0
        self.deleted_ids = deque()

    def update(self, overlap_objects: OverlapDetector):
        self.deleted_ids = deque()
        overlap_objects_list = deque()
        for ool in overlap_objects:
            if ool[3] != -1:
                overlap_objects_list.append(DetectedObject(ool[0], ool[1], ool[2], ool[3], ool[4]))

        to_remove_index = []
        for to in self.tracked_objects:
            if to in overlap_objects_list:
                index_of_match = overlap_objects_list.index(to)
                to.update(overlap_objects_list[index_of_match])
                del overlap_objects_list[index_of_match]
            else:
                if to.missing():
                    self.deleted_ids.append(to.tracker_id)
                    to_remove_index.append(self.tracked_objects.index(to))

        for to_remove in to_remove_index:
            if to_remove < len(self.tracked_objects):
                del self.tracked_objects[to_remove]

        if len(overlap_objects_list):
            self.tracked_objects.extend(overlap_objects_list)
        self.merge_tensors()

    def get_statistics(self):
        valid_objects = [o for o in self.tracked_objects if o.uptime > 20]
        total = len(valid_objects)
        without_overlap = 0
        changed = False
        for to in valid_objects:
            if not to.get()[4]:
                without_overlap += 1
        if without_overlap != self.without or self.total != total:
            changed = True

        self.without = without_overlap
        self.total = total

        return (without_overlap, total, changed)

    def merge_tensors(self):
        self.filtered_tracked_objects = [to for to in self.tracked_objects if not to.missing()]

        self.bbox = np.array([to.bbox for to in self.filtered_tracked_objects])
        self.confidence = np.array([to.confidence for to in self.filtered_tracked_objects])
        self.class_id = np.array([to.class_id for to in self.filtered_tracked_objects])
        self.tracker_id = np.array([to.tracker_id for to in self.filtered_tracked_objects])

    def __len__(self):
        return len(self.filtered_tracked_objects)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int, float, int]]:
        for o in self.filtered_tracked_objects:
            yield o.get()

    def get_deleted_objects(self):
        return self.deleted_ids

    def set_settings(settings):
        if "EVENT_DETECTION" in settings:
            for key, val in settings["EVENT_DETECTION"].items():
                print(f"{key} = {val}")
                exec(key + "=val")
