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

from collections import deque
from datetime import datetime


class BaseState:
    def __init__(self, fsm):
        self.init_run = True
        self.fsm = fsm
        self.frames = deque()
        self.start_time = None
        self.end_time = None

    def run(self, frame):
        if self.init_run:
            self.init_run = False
            self.reset_frames()
            self.start_time = datetime.fromisoformat(frame["T"])

        label = frame["label"].upper()
        mapped_class = self.fsm.map[label] if label in self.fsm.map else "OUT OF RANGE"
        if label in self.fsm.map:
            pass
        else:
            print(f"{label} not in {self.fsm.map}")

        frame["mapped_class"] = mapped_class
        return mapped_class

    def leave(self, frame):
        self.init_run = True
        self.end_time = datetime.fromisoformat(frame["T"])

    def store_frame(self, frame, max_length=600):
        self.frames.append(frame)
        if len(self.frames) > max_length:
            self.frames.popleft()

    def reset_frames(self):
        self.frames.clear()

    def change_state_to(self, new_state, frame):
        self.fsm.change_state_to(new_state)
        self.leave(frame)

    def get_state_settings(self):
        state_name = self.__class__.__name__
        if state_name in self.fsm.settings["Fsm"]["States"]:
            return self.fsm.settings["Fsm"]["States"][state_name]
        else:
            return None

    def get_settings(self):
        return self.fsm.settings

    def get_time_in_state(self, frame):
        start = self.start_time
        end = datetime.fromisoformat(frame["T"])

        elapsed_time = end - start
        return elapsed_time.total_seconds()

    def get_availability_time_threshold(self):
        settings = self.fsm.settings
        return (
            settings["Fsm"]["Production"]["TAKT_TIME"]
            * settings["Fsm"]["Production"]["AVAILABILITY_FACTOR"]
        )
