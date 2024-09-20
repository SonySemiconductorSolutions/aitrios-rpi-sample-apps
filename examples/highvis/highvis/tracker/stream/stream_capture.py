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


class StreamCapture:
    def __init__(self, source):
        self.source = source
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()

    def __iter__(self):
        self.last_time = time.time()
        self.source.__iter__()
        return self

    def __next__(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        self.source.FPS = self.calculate_fps()
        return self.source.__next__()

    def __enter__(self):
        self.source.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.source.__exit__(exc_type, exc_value, traceback)

    def calculate_fps(self):
        if len(self.frame_times) > 1:
            total_time = sum(self.frame_times)
            fps = len(self.frame_times) / total_time
            return fps
        else:
            return 0.0


class StreamCaptureAsync:
    def __init__(self, source):
        self.source = source
        self.frame_times = deque(maxlen=30)
        self.last_time = time.time()

    def __aiter__(self):
        self.last_time = time.time()
        self.source.__aiter__()
        return self

    async def __anext__(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)
        self.last_time = current_time
        self.source.FPS = self.calculate_fps()
        return await self.source.__anext__()

    async def __aenter__(self):
        await self.source.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.source.__aexit__(exc_type, exc_value, traceback)

    def calculate_fps(self):
        if len(self.frame_times) > 1:
            total_time = sum(self.frame_times)
            fps = len(self.frame_times) / total_time
            return fps
        else:
            return 0.0
