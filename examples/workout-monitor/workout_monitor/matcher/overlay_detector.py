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

import numpy as np

# from visionary.core.tracker.bbox_overlaps import bbox_overlaps
from workout_monitor.tracker import BYTETracker, STrack, Detections
from typing import List, Tuple, Iterator, Union


def bbox_intersect_ratio(atlbrs, btlbrs):
    """
    Compute the Ratio of Intersection between two sets of bounding boxes.

    :param atlbrs: An array of shape (M, 4) representing the first set of bounding boxes.
    :param btlbrs: An array of shape (N, 4) representing the second set of bounding boxes.
    :return: A matrix of shape (M, N) where element (i, j) is the bboxir between the ith bounding box in atlbrs and the jth bounding box in btlbrs.
    """

    # Expand dimensions to allow broadcasting
    a_areas = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    b_areas = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    # Broadcast the bounding boxes' coordinates for easier computation
    inter_x1 = np.maximum(atlbrs[:, None, 0], btlbrs[:, 0])
    inter_y1 = np.maximum(atlbrs[:, None, 1], btlbrs[:, 1])
    inter_x2 = np.minimum(atlbrs[:, None, 2], btlbrs[:, 2])
    inter_y2 = np.minimum(atlbrs[:, None, 3], btlbrs[:, 3])

    # Compute the area of intersection rectangle
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Compute the area of union
    union_area = (a_areas[:, None] + b_areas) - inter_area

    overlap_ratio = inter_area / b_areas

    return overlap_ratio


class OverlapDetector(Detections):
    def __init__(self, detections: Detections, base_object_class: int, overlayed_objects_class: int, settings: dict):
        super().__init__(detections.frame_size[0], detections.frame_size[0])
        self.detections = detections

        self.base_object_class = base_object_class
        self.overlayed_objects_class = overlayed_objects_class

        if "OVERLAP_DETECTION" in settings:
            self.modified_bbox = settings["OVERLAP_DETECTION"]["MODIFIED_BBOX"]
            self.box_size = settings["OVERLAP_DETECTION"]["OVERLAP_BOX_SIZE_PX"]
        else:
            self.modified_bbox = False
            self.box_size = 15

        self.bboxir = np.empty((0,))
        self.maximum_bboxir = 0

    def filter(self):
        mask = self.detections.class_id == self.base_object_class
        mask_overlay = self.detections.class_id == self.overlayed_objects_class

        self.confidence = self.detections.confidence[mask]
        self.tracker_id = self.detections.tracker_id[mask]
        self.base_bboxes = self.detections.bbox[mask]

        self.overlay_bboxes = self.detections.bbox[mask_overlay]

        self.bbox = self.detections.bbox[mask]
        self.class_id = self.detections.class_id[mask]

        if len(self.base_bboxes) and len(self.overlay_bboxes):
            self.bboxir = bbox_intersect_ratio(self.base_bboxes, self.overlay_bboxes)
        elif len(self.base_bboxes):
            self.bboxir = np.zeros((len(self.base_bboxes), 1))
        else:
            return

        if self.modified_bbox:
            self.modify_bbox()

    def __len__(self):
        # if len(self.base_bboxes) and len(self.overlay_bboxes):
        if len(self.base_bboxes):
            self.maximum_bboxir = np.argmax(self.bboxir, axis=1)
            return len(self.bboxir)
        else:
            return 0

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int, float]]:
        for i in range(len(self)):
            yield (self.bbox[i], self.confidence[i], self.class_id[i], self.tracker_id[i], self.bboxir[i, self.maximum_bboxir[i]])

    def modify_bbox(self):

        centers = find_bbox_center_point(self.bbox)
        bboxes = create_bbox_from_point(centers, self.box_size)
        self.bbox = bboxes


def create_bbox_from_point(center_points, size):

    new_X = (center_points[:, 0]) - size / 2
    new_x = (center_points[:, 0]) + size / 2
    new_Y = (center_points[:, 1]) - size / 2
    new_y = (center_points[:, 1]) + size / 2

    bboxes = np.column_stack((new_X, new_Y, new_x, new_y))
    return bboxes


def find_bbox_center_point(bboxes):
    X = bboxes[:, 0]
    x = bboxes[:, 2]
    Y = bboxes[:, 1]
    y = bboxes[:, 3]

    center_x = (X + x) / 2
    center_y = (Y + y) / 2
    center_points = np.column_stack((center_x, center_y))
    return center_points
