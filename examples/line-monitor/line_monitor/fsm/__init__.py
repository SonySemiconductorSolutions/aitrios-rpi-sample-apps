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

from .fsm import FiniteStateMachine
import logging

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)

DEFAULT_SETTINGS = {
    "Map": {
        "BACKGROUND": "BACKGROUND",
        "GOOD": "OBJECT",
        "BAD": "OBJECT",
        "GOOD": "RESULT_OK",
        "BAD": "RESULT_BAD",
        "EMPTY": "BACKGROUND",
        "LACK": "RESULT_BAD",
    },
    "Fsm": {
        "States": {
            "StateInit": {},
            "StateWaitForBackGround": {"MINIMUM_TRANSITION_FRAMES": 10},
            "StateWaitForPanel": {"MINIMUM_TRANSITION_FRAMES": 5},
            "StateScanning": {"MINIMUM_SCAN_FRAMES": 5, "MAXIMUM_SCAN_FRAMES": 20},
            "StateCalculate": {"PROBABILITY_THRESHOLD_PERCENT": 90},
        },
        "Production": {"TAKT_TIME": 4, "AVAILABILITY_FACTOR": 1.2},
    },
    "AI": {"Model": "networks/imx500_network_solderpoints.rpk", "Labels": "networks/labels.txt"},
}
