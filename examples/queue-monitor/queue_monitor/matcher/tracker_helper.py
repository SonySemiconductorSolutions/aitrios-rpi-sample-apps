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
from typing import List

from queue_monitor.tracker import BYTETracker, STrack, Detections
from queue_monitor.tracker.tracker.bbox_overlaps import bbox_overlaps


class BYTETrackerArgs:

    def __init__(self, settings=None):
        if "BYTE_TRACKER" in settings:
            for key, val in settings["BYTE_TRACKER"].items():
                exec(f"self.{key.lower()} =val")
                print(f"{key.lower()} -> {val}")

    def __str__(self):
        ret = f"track_thresh {self.track_thresh}, track_buffer {self.track_buffer}, match_thresh {self.match_thresh},aspect_ratio_thresh {self.aspect_ratio_thresh},min_box_area {self.min_box_area}, mot20{self.mot20}"
        return ret


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(detections: Detections, tracks: List[STrack]) -> np.array:
    """Match the current detections with tracker predicted objects."""
    if not np.any(detections) or len(tracks) == 0:
        # Keep the tracker_ids array length in sync with bbox len to avoid crash further down.
        return np.full(len(detections), -1)
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = bbox_overlaps(tracks_boxes, detections)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = np.array([-1] * len(detections))

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids
