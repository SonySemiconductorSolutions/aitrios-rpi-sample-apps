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
import os
from datetime import datetime, timedelta
import logging

from libcamera import Rectangle, Size
from picamera2 import MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500

from .frame_rate import FrameRate
from .worker import AbstractWorker

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)


class IMX500Data:
    def __init__(self, configuration: dict, callback: AbstractWorker):
        self.configuration = configuration

        # This must be called before instantiation of Picamera2
        self.imx500 = IMX500(network_file=self.model_path())
        self.imx500.set_inference_aspect_ratio(self.imx500.config["input_tensor_size"])

        self.picam2 = Picamera2()
        self.config = self.picam2.create_preview_configuration(controls={"FrameRate": 30})
        self.callback = callback
        self.show_preview = True
        self.boot_time_duration_seconds = time.clock_gettime(time.CLOCK_BOOTTIME)
        self.frame_rate = FrameRate(filter_length=30)

    def fps(self):
        return self.frame_rate.calc()

    def start_stream(self):
        self.picam2.start(self.config, show_preview=self.show_preview)
        data = {}
        for _ in range(100):
            try:
                data["meta_data"] = self.picam2.capture_metadata()
                data["camera_configuration"] = self.picam2.camera_configuration()
                data["camera_properties"] = self.picam2.camera_properties
                if not "Imx500OutputTensorInfo" in data["meta_data"]:
                    continue
                break
            except KeyError:
                logging.info("Waiting for data...")
                time.sleep(0.6)
                pass
        self.picam2.pre_callback = self.picam_callback
        return data

    def stop_stream(self):
        self.picam2.stop_preview()
        self.picam2.stop()

    def picam_callback(self, request):

        imx_tensors = {}
        self.frame_rate.add()
        imx_tensors["Imx500OutputTensor"] = request.get_metadata().get("CnnOutputTensor")
        imx_tensors["Imx500InputTensor"] = request.get_metadata().get("CnnInputTensor")
        imx_tensors["CaptureTime"] = self.convert_timestamp_to_capture_time(request.get_metadata().get("SensorTimestamp"))

        if not imx_tensors["Imx500OutputTensor"] == None:
            with MappedArray(request, "main") as m:
                self.callback.do_work_callback(imx_tensors, m.array)

    def model_path(self):
        model_fpk_path = os.path.abspath(self.configuration["AI"]["Model"])
        logging.info(f"Loading model:  {model_fpk_path}")
        return model_fpk_path

    def convert_timestamp_to_capture_time(self, sensor_timestamp):

        current_time_seconds = time.time()
        boot_time_seconds = current_time_seconds - self.boot_time_duration_seconds
        boot_time = datetime.fromtimestamp(boot_time_seconds)
        capture_time = boot_time + timedelta(microseconds=sensor_timestamp / 1000)

        return capture_time
