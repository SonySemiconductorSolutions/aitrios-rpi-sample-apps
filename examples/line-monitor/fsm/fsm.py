#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from . import states as s

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


class FiniteStateMachine:
    def __init__(self, settings):
        self.settings = settings
        self.map = self.settings["Map"]
        self.states = {
            "StateInit": s.StateInit(self),
            "StateWaitForBackGround": s.StateWaitForBackGround(self),
            "StateWaitForPanel": s.StateWaitForPanel(self),
            "StateScanning": s.StateScanning(self),
            "StateCalculate": s.StateCalculate(self),
        }
        self.current_state = "StateInit"
        self.previous_state = "StateInit"
        self.scan_seq = None
        self.last_result = {}
        self.last_frame = None
        self.n_tick = 0

    def change_state_to(self, new_state):
        if new_state in self.states:
            logging.info("********************************************************")
            logging.info(f"Changed state from {self.current_state} to {new_state}.")
            self.previous_state = self.current_state
            self.current_state = new_state

    def tick(self, frame):
        self.n_tick += 1
        frame["state"] = self.current_state
        result = self.states[self.current_state].run(frame)
        if result and "scan_result" in result:
            self.last_result = result
            if "SUBCLASS" in result["scan_result"]:
                print("### TODO", result["scan_result"]["SUBCLASS"])
                # result["scan_result"]["SUBCLASS_LABEL"] = self.mapper.class_to_label(result["scan_result"]["SUBCLASS"])
        self.last_frame = frame
        return result

    def info(self):
        ret = [
            f"C: {self.last_frame['label']}",
            f"P: {int(self.last_frame['P'] * 100)}",
            f"T: {self.last_frame['T']}",
            f"M: {self.last_frame['mapped_class']}",
            f"State: {self.current_state}",
        ]
        return ret

    def __str__(self):
        ret = "\n".join(str(self))
        return ret
