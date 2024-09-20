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

from queue import Queue, Empty
import time

import cv2
import numpy as np
from libcamera import Rectangle, Size
from picamera2 import MappedArray, Picamera2

from collections import deque


class IMX500Receiver:
    def __init__(self, show_preview=True):
        self.queue = Queue()
        self.work_queue = Queue()
        self.frame_times = deque(maxlen=5)

        self.frame_number = 0
        self.last_time = 0

        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(controls={"FrameRate": 30})
        self.picam2.pre_callback = self.picam_callback

        self.show_preview = show_preview

    def start_stream(self):
        self.picam2.start(self.config, show_preview=self.show_preview)
        data = {}
        for _ in range(10):
            try:
                data["meta_data"] = self.picam2.capture_metadata()
                data["camera_configuration"] = self.picam2.camera_configuration()
                data["camera_properties"] = self.picam2.camera_properties

                break
            except KeyError:
                pass

        return data

    def picam_callback(self, request):
        fps = self._get_fps()

        imx_tensor = {}
        imx_tensor["Imx500OutputTensor"] = request.get_metadata().get("Imx500OutputTensor")
        imx_tensor["Imx500InputTensor"] = request.get_metadata().get("Imx500InputTensor")
        imx_tensor["FrameNumber"] = self.frame_number

        self.queue.put(imx_tensor)

        with MappedArray(request, "main") as m:
            self._annotate_frame_info(m.array, fps, self.frame_number)

            try:
                # Wait a short while (no longer then frame period) to allow the highvis calculation to finish.
                # Data in the workin item has not been deep copied (TODO) so only work on the last frame)
                item = self.work_queue.get(timeout=0.015)
                item.annotate(m.array)
                while not self.work_queue.empty():
                    print(f"Flushing queue fps: {fps}")
                    self.work_queue.get_nowait()
            except Empty:
                print(f"Queue empty exception.")

        self.frame_number += 1

    def fifo_pop(self, timeout=5):
        return self.queue.get(timeout=timeout)

    def annotation_queue_put(self, work):
        self.work_queue.put(work)

    def _get_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time - self.last_time)

        self.last_time = current_time
        if len(self.frame_times) > 1:
            total_time = sum(self.frame_times)
            fps = len(self.frame_times) / total_time
            return fps
        else:
            return 0.0

    def _annotate_frame_info(self, scene: np.ndarray, fps: float, frame_number: int):
        cv2.putText(
            img=scene,
            text=f"{frame_number: 8d} fps: {fps:2.1f}",
            org=(0, 25),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 50, 50),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return scene
