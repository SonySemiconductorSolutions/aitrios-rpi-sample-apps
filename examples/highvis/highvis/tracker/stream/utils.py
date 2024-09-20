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
import os

import cv2
import numpy as np
from libcamera import Rectangle, Size
from picamera2 import MappedArray, Picamera2

from picamera2.devices.imx500 import IMX500
from picamera2.devices.imx500.postprocess_nanodet import postprocess_nanodet_detection

from collections import deque

last_detection = None


class IMX500Receiver:
    def __init__(self, configuration, show_preview=True):
        self.configuration = configuration
        self.queue = Queue()
        self.work_queue = Queue()
        self.frame_times = deque(maxlen=5)

        self.frame_number = 0
        self.last_time = 0
        self.imx500 = IMX500(network_file=self.model_path())

        # self.imx500.set_auto_aspect_ratio()

        self.picam2 = Picamera2()

        self.config = self.picam2.create_preview_configuration(controls=configuration["PICAMERA_CONTROLS"])
        self.picam2.pre_callback = self.picam_callback

        self.show_preview = show_preview

    def model_path(self):
        model_fpk_path = os.path.abspath(self.configuration[self.configuration["IMX_CONFIG_SELECTOR"]]["NETWORK_PATH"])
        print(f"Loading model: {model_fpk_path}")
        return model_fpk_path

    def start_stream(self):
        self.picam2.start(self.config, show_preview=self.show_preview)

    def decode_meta_data(self, metadata):
        global last_detection

        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)

        if np_outputs is None:
            return last_detection

        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]

        new_bbox = np.empty((0, 4))

        for row in boxes:
            scaled_box = self.imx500.convert_inference_coords(row, metadata, self.picam2)
            absolute_box = np.array([scaled_box.x, scaled_box.y, scaled_box.x + scaled_box.width, scaled_box.y + scaled_box.height])
            new_bbox = np.vstack([new_bbox, absolute_box])

        bbox = new_bbox

        ret = np.column_stack((bbox, scores, classes))
        last_detection = ret

        return ret

    def picam_callback(self, request):
        fps = self._get_fps()
        output_tensor = self.decode_meta_data(request.get_metadata())

        if type(output_tensor) == np.ndarray:
            self.queue.put(output_tensor)

        with MappedArray(request, "main") as m:
            self._annotate_frame_info(m.array, fps, self.frame_number)

            try:
                # Wait a short while (no longer then frame period) to allow the highvis calculation to finish.
                item = self.work_queue.get(timeout=0.015)
                item.annotate(m.array)
                while not self.work_queue.empty():
                    self.work_queue.get_nowait()
            except Empty:
                pass

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
