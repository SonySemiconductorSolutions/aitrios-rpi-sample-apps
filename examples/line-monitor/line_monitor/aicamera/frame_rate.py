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

import time
from collections import deque


class FrameRate:
    def __init__(self, filter_length=30):
        self.timestamps = deque(maxlen=filter_length)
        self.last_time = time.time()
        self.filter_lenght = filter_length

    def add(self):
        current_time = time.time()
        self.timestamps.append(current_time - self.last_time)
        self.last_time = current_time

    def calc(self):
        if len(self.timestamps) == self.filter_lenght:
            total_time = sum(self.timestamps)
            fps = len(self.timestamps) / total_time
            return fps
        else:
            return -1
