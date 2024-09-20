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

import signal
import cv2
import numpy as np

from highvis.tracker import Detections, BYTETracker, StreamCapture, IMX500, BoxAnnotator, ColorPalette
from highvis.matcher import OverlapDetector, ObjectEventLogic, settings, shutdown_handler, WorkQueueAnnotator, tracker_helper, an_colors


class HighvisExample:
    def __init__(self):
        self.settings_dict = settings("settings.json")

        ObjectEventLogic.set_settings(self.settings_dict)

        signal.signal(signal.SIGINT, shutdown_handler)

        self.source = IMX500(self.settings_dict)

        # Initialize the tracker, this layer will track an object over time. Each object will be assigned a tracker id.
        self.tracker = BYTETracker(tracker_helper.BYTETrackerArgs(self.settings_dict))
        self.detections = Detections(frame_height=self.source.height, frame_width=self.source.width)

        # The object logic layer is filtering and providing stable results for the the matched overlayed objects. Thresholds and hysteresis.
        self.object_logic = ObjectEventLogic(self.detections)

        # Initialization of the annotation functions for each layer.
        self.annotators = [
            BoxAnnotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.2),
            BoxAnnotator(color=ColorPalette.from_hex(an_colors[0]), thickness=1, text_thickness=1, text_scale=0.5),
            BoxAnnotator(color=ColorPalette.from_hex(an_colors[1]), thickness=1, text_thickness=1, text_scale=0.5),
        ]

    def run(self):

        labels = self.settings_dict["IMX_CONFIG"]["LABELS"]

        with StreamCapture(self.source) as stream:
            next(stream)
            self.detections.frame_size = (self.source.height, self.source.width)

            for frame in stream:

                # Update detections with resulting IMX500 output tensor
                if frame == None:
                    continue
                self.detections.update(frame.result)

                # Filter out the detected objects that are above the confidence threshold.
                self.detections = self.detections[self.detections.confidence > self.settings_dict["CONFIDENCE_THRESHOLD"]]

                # Tick the tracker with data from the latest frame.
                tracks = self.tracker.update(self.detections.to_tracker())

                self.detections.tracker_id = tracker_helper.match_detections_with_tracks(self.detections, tracks)
                detection = self.detections.copy()

                # Detect which objects of type "Person" and "Vest" that are overlapped.
                overlap_detector = OverlapDetector(detection, labels.index("Person"), labels.index("Vest"), self.settings_dict)
                overlap_detector.filter()

                # Do the logic to determain the current state of objects over lapped or not. This layer helps stablizing the results.
                self.object_logic.update(overlap_detector)
                (missing_vest, total, changed) = self.object_logic.get_statistics()

                # If status changed of detected people w/wo vest, print the new status.
                if changed:
                    print(f"Total People: {total} Missing vest: {missing_vest}.")

                # Gather all the detectors used in the pipeline to and use this to annotate the high resolution image.
                detectors = [detection, overlap_detector, self.object_logic]
                self.source.work_put(
                    WorkQueueAnnotator(
                        detectors,
                        self.annotators,
                        self.settings_dict,
                    )
                )


def start_highvis_demo():
    cv2.startWindowThread()
    hv = HighvisExample()
    hv.run()


if __name__ == "__main__":
    start_highvis_demo()
    exit()
