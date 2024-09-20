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

import os
import signal
import sys
from datetime import datetime
from time import time
import json
from queue_monitor.matcher.default_settings import SETTINGS_DEFAULT


SSD_MOBILENET_CLASSES = ["Person", "Vest"]
NANODET_CLASSES = ["Nothing", "Person", "Vest"]

CLASSES = None  # NANODET_CLASSES
settings_global = None


def frame_annotate(detectors, frame, annotators):
    # Generate labels for annotation.
    global settings_global
    global CLASSES
    an_setting = settings_global["ANNOTATION"]

    if an_setting["BYTE_TRACKER"]:
        labels = []
        for bbox, confidence, class_id, tracker_id in detectors[0]:
            labels += [f"#{tracker_id} {CLASSES[class_id]} {confidence:0.2f}"]
        frame = annotators[0].annotate(scene=frame, detections=detectors[0], labels=labels)

    if an_setting["OBJECT_OVERLAY"]:
        iou_text = []
        for bbox, confidence, class_id, tracker_id, iou in detectors[1]:
            iou_text += [f"#{tracker_id} OL:{iou:0.2f}"]
        frame = annotators[1].annotate(scene=frame, detections=detectors[1], labels=iou_text)

    if an_setting["OBJECT_EVENT"]:
        logic_text = []
        for bbox, avg_overlap, class_id, tracker_id, overlap, uptime in detectors[2]:
            logic_text += [f"P:{tracker_id} {overlap}:{avg_overlap:0.1f} {uptime}"]
        frame = annotators[2].annotate(scene=frame, detections=detectors[2], labels=logic_text)

    return frame


def settings(filename):
    global CLASSES
    global settings_global
    if os.path.exists(filename):
        # File exists, read JSON content
        with open(filename, "r") as file:
            try:
                settings = json.load(file)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}. Using default content.")
                settings = SETTINGS_DEFAULT
    else:
        # File doesn't exist, create and write default content
        with open(filename, "w") as file:
            settings = SETTINGS_DEFAULT or {}
            json.dump(settings, file, indent=2)
            print(f"File {filename} created with default content.")
    settings["IMX_CONFIG"] = settings[settings["IMX_CONFIG_SELECTOR"]]
    CLASSES = settings["IMX_CONFIG"]["LABELS"]
    settings_global = settings

    print(f"Got settings: {json.dumps(settings_global, indent=1)}")
    print(f"Got labels: {CLASSES}")
    return settings


def timestamp():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d_%H%M%S")


def shutdown_handler(signum, frame):
    """
    Signal handler for graceful shutdown.
    """
    print("Received signal for shutdown. Cleaning up...")
    # Perform cleanup operations here if needed
    sys.exit(0)


class frame_saver:
    SAVE_N_FRAMES = 10
    frame_counter = 0
    sequence_counter = 0
    frames_left_to_save = 1
    timestamp_start = ""

    def save_frame(frame, changed, missing, total):
        import cv2

        if frame_saver.frames_left_to_save > 0:
            if frame_saver.frame_counter == 0:
                frame_saver.timestamp_start = timestamp()
            filename = f"./images/ts{frame_saver.timestamp_start}_{frame_saver.sequence_counter:05}_{frame_saver.frame_counter:05}_missing{missing}_tot{total}.jpeg"

            cv2.imwrite(filename, frame)
            frame_saver.frames_left_to_save -= 1
            frame_saver.frame_counter += 1
            if not frame_saver.frames_left_to_save:
                print(f"Saved seq:{frame_saver.sequence_counter} n frames:{frame_saver.frame_counter} Last file: {filename}")
                frame_saver.frame_counter = 0
                frame_saver.sequence_counter += 1

        if total:
            frame_saver.frames_left_to_save = frame_saver.SAVE_N_FRAMES
