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
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)

SETTINGS_DEFAULT = {
    "IMX_CONFIG_NANODET": {
        "NETWORK_PATH": "networks/imx500_network_nanodet_ppe.rpk",
        "SWAP_TENSORS": False,
        "LABELS": ["safety-equipment", "Person", "goggles", "helmet", "no-goggles", "no-helmet", "no-vest", "Vest"],
    },
    "FRAME_RATE": 30,
    "PICAMERA_CONTROLS": {"FrameRate": 30},
    "IMX_CONFIG_SELECTOR": "IMX_CONFIG_NANODET",
    "BYTE_TRACKER": {
        "TRACK_THRESH": 0.60,
        "TRACK_BUFFER": 240,
        "MATCH_THRESH": 0.6,
        "ASPECT_RATIO_THRESH": 3.0,
        "MIN_BOX_AREA": 1.0,
        "MOT20": False,
    },
    "CONFIDENCE_THRESHOLD": 0.6,
    "OVERLAP_DETECTION": {"OVERLAP_BOX_SIZE_PX": 20, "MODIFIED_BBOX": False},
    "EVENT_DETECTION": {"MAX_MISSING_OVERLAP": 120, "MAX_MISSING_TRACKER": 60, "MIN_OVERLAP_THRESHOLD": 0.5, "HYSTERESIS": 0.4},
    "CLOUD": False,
    "SAVE_IMAGE": False,
    "EMIT_DETAILED_RESULT": True,
    "MINIMUM_UPTIME": 15,
    "ANNOTATION": {"BYTE_TRACKER": False, "OBJECT_OVERLAY": False, "OBJECT_EVENT": True},
}


SSD_MOBILENET_CLASSES = ["Person", "Vest"]
NANODET_CLASSES = ["Nothing", "Person", "Vest"]

CLASSES = None
settings_global = None


def settings(filename):
    global CLASSES
    global settings_global
    if os.path.exists(filename):
        # File exists, read JSON content
        with open(filename, "r") as file:
            try:
                settings = json.load(file)
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from {filename}. Using default content.")
                settings = SETTINGS_DEFAULT
    else:
        # File doesn't exist, create and write default content
        with open(filename, "w") as file:
            settings = SETTINGS_DEFAULT or {}
            json.dump(settings, file, indent=2)
            logging.info(f"File {filename} created with default content.")
    settings["IMX_CONFIG"] = settings[settings["IMX_CONFIG_SELECTOR"]]
    CLASSES = settings["IMX_CONFIG"]["LABELS"]
    settings_global = settings

    logging.info(f"Got settings: {json.dumps(settings_global, indent=1)}\nLabels: {{CLASSES}}")
    return settings


def timestamp():
    current_datetime = datetime.now()
    return current_datetime.strftime("%Y-%m-%d_%H%M%S")


def shutdown_handler(signum, frame):
    """
    Signal handler for graceful shutdown.
    """
    logging.warning("Received signal for shutdown. Cleaning up...")
    # Perform cleanup operations here if needed
    sys.exit(0)
