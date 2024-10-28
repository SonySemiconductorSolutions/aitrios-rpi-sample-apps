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
from typing import List, Tuple, Iterator, Union
from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess_yolov8 import postprocess_yolov8_detection
from picamera2.devices.imx500.postprocess import scale_boxes
from parking_monitor.tracker import BYTETracker
from parking_monitor.utils.tracker_helper import match_detections_with_tracks, BYTETrackerArgs


class ToTrack:
    output_results: np.ndarray
    img_info: Tuple[int, int]
    img_size: Tuple[int, int]


SETTINGS = {
    "BYTE_TRACKER": {
        "TRACK_THRESH": 0.30,
        "TRACK_BUFFER": 240,
        "MATCH_THRESH": 0.68,
        "ASPECT_RATIO_THRESH": 3.0,
        "MIN_BOX_AREA": 1.0,
        "MOT20": False,
    }
}


class Detections:
    """
    Attributes:
    - bbox (np.ndarray):            Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
    - confidence (np.ndarray):      Array of shape (n,) the confidence of N detections
    - class_id (np.ndarray):        Array of shape (n,) the class id of N detections
    - tracker_id (np.ndarray):      Array of shape (n,) the tracker id of N detections

    Properties:
    - area (np.ndarray):            Array of shape (n,) the area of the bounding boxes of N detections
    - bbox_width (np.ndarray):      Array of shape (n,) the width of the bounding boxes of N detections
    - bbox_height (np.ndarray):     Array of shape (n,) the height of the bounding boxes of N detections
    """

    bbox: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: np.ndarray
    frame_size: Tuple[int, int]

    def __init__(self, device) -> None:
        self.imx500 = device
        self.bbox = np.empty((0, 4))
        self.confidence = np.empty((0,))
        self.class_id = np.empty((0,))
        self.tracker_id = np.empty((0,))
        self.BOX_MIN_CONFIDENCE = 0.3
        self.IOU_THRESHOLD = 0.7
        self.MAX_OUT_DETS = 300
        self.tracker = BYTETracker(BYTETrackerArgs(SETTINGS))

    def update(self, metadata: dict):
        """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
        global last_boxes, last_scores, last_classes
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is not None:
            last_boxes, last_scores, last_classes = [], [], []
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            for box, score, category in zip(boxes, scores, classes):
                if score > self.BOX_MIN_CONFIDENCE:
                    last_boxes.append(box)
                    last_scores.append(score)
                    last_classes.append(category)
        self.bbox = np.array(last_boxes)
        self.confidence = np.array(last_scores)
        self.class_id = last_classes
        return np.array(last_boxes), np.array(last_scores), last_classes

    def format_track(self):
        """Format and get the tracker IDs for the detected objects"""
        out = ToTrack()
        out.img_info = [640, 640]
        out.img_size = [640, 640]
        out.output_results = np.hstack((np.array(self.bbox), self.confidence[:, np.newaxis]))

        tracks = self.tracker.update(out)
        tracker_id = match_detections_with_tracks(self.bbox, tracks)
        return tracker_id

    ## OPERATORS
    def __len__(self):
        """
        Return the number of detections.
        """
        return len(self.bbox)

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Detections":
        """
        Returns a new Detections object with the selected detections.
        Could be a subsection of the current detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.bbox = self.bbox[index]
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.tracker_id = self.tracker_id[index]
        return res

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int]]:
        for i in range(len(self)):
            yield (
                self.bbox[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i],
            )

    ## LOAD SOURCES
    def clear(self):
        """
        Clear the current detections.
        """
        self.bbox = np.empty((0, 4))
        self.confidence = np.empty((0,))
        self.class_id = np.empty((0,))
        self.tracker_id = np.empty((0,))
