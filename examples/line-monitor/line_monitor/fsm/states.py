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

import logging
from .scan_sequence import ScanSequence as ss
from .base_state import BaseState

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)

LINE_STATUS = {"INIT": 0, "PRODUCING": 1, "NOT_PRODUCING": 10, "BLOCKED": 11, "STARVED": 12}


def create_availability_result(status, frame, elapsed_time):
    result = {}
    result["line_info"] = {}
    result["line_info"]["STATE"] = LINE_STATUS[status]
    result["line_info"]["TEXT"] = status
    result["line_info"]["T"] = frame["T"].isoformat()
    result["line_info"]["ELAPSED_TIME"] = elapsed_time
    return result


class StateInit(BaseState):
    def run(self, frame):
        super().run(frame)
        self.change_state_to("StateWaitForBackGround", frame)
        return None


class StateWaitForBackGround(BaseState):
    def run(self, frame):
        mapped_class = super().run(frame)

        if ss.class_is_background(mapped_class):
            self.change_state_to("StateWaitForPanel", frame)

        elapsed_time = self.get_time_in_state(frame)
        block_time = self.get_availability_time_threshold()
        if elapsed_time > block_time:
            self.change_state_to("StateWaitForBackGround", frame)  # Reset
            return create_availability_result("BLOCKED", frame, elapsed_time)
        return None


class StateWaitForPanel(BaseState):
    def run(self, frame):
        mapped_class = super().run(frame)

        if not ss.class_is_background(mapped_class):
            self.store_frame(frame, self.get_state_settings()["MINIMUM_TRANSITION_FRAMES"] + 1)
            if len(self.frames) > self.get_state_settings()["MINIMUM_TRANSITION_FRAMES"]:
                self.change_state_to("StateScanning", frame)
        else:
            self.reset_frames()

        elapsed_time = self.get_time_in_state(frame)
        starve_time = self.get_availability_time_threshold()
        if elapsed_time > starve_time:
            self.change_state_to("StateWaitForPanel", frame)  # Reset
            return create_availability_result("STARVED", frame, elapsed_time)
        return None


class StateScanning(BaseState):
    def run(self, frame):

        mapped_class = super().run(frame)

        if not ss.class_is_background(mapped_class):
            self.store_frame(frame)
            if len(self.frames) > self.get_state_settings()["MAXIMUM_SCAN_FRAMES"]:
                print("")
                self.fsm.scan_seq = ss(self.frames, self.fsm.last_result)
                self.change_state_to("StateCalculate", frame)
                return create_availability_result("PRODUCING", frame, 0)
            else:
                character = "G" if frame["mapped_class"] == "RESULT_OK" else "B"
                print(f"{character}", end=" ", flush=True)
        else:
            if len(self.frames) > self.get_state_settings()["MINIMUM_SCAN_FRAMES"]:
                self.fsm.scan_seq = ss(self.frames, self.fsm.last_result)
                self.change_state_to("StateCalculate", frame)
                return create_availability_result("PRODUCING", frame, 0)
            else:
                self.store_frame(frame)
                logging.info(
                    f"Forced to stay in StateScanning for: {len(self.frames)} / {self.get_state_settings()['MINIMUM_SCAN_FRAMES']}"
                )

        return None


class StateCalculate(BaseState):
    def run(self, frame):
        super().run(frame)
        result = self.fsm.scan_seq.analyze(self.get_state_settings())
        logging.info(f"Result after analysis {result}")
        self.change_state_to("StateWaitForBackGround", frame)
        return result

        # exit()
