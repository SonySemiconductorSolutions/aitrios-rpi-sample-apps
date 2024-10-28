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

import base64
import numpy as np
from abc import ABC, abstractmethod
import struct
from picamera2.devices.imx500 import IMX500
from libcamera import Rectangle, Size
import json


class AbstractSource(ABC):
    """
    Abstract base class for frame sources.

    This class defines a common interface for various types of frame sources.
    Subclasses should provide concrete implementations for the methods defined here.

    Attributes:
        width (int): Width of the frames provided by the source.
        height (int): Height of the frames provided by the source.
    """

    def __init__(self):
        self.width = None
        self.height = None
        self.FPS = 0

    @abstractmethod
    def get_frame(self):
        """
        Abstract method to retrieve the next frame from the source.
        This method must be overridden by subclasses to return the next available frame.

        Returns:
            The next frame as an image array or None if no more frames are available.
        """
        pass

    @abstractmethod
    def __iter__(self):
        """
        Iterator. Subclasses should implement this method to allow iteration over the source.

        Returns: Nothing! To keep the StreamCapture as main iterator
        """
        pass

    @abstractmethod
    def __next__(self):
        """
        Return the next item from the source.
        Subclasses should implement this method to provide the next frame.

        Returns:
            The next frame as an image array.

        Raises:
            StopIteration: When no more frames are available.
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Abstract method to perform any setup when entering the runtime context.
        This may involve establishing connections.
        Returns: Nothing
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Abstract method to perform any cleanup when exiting the runtime context.
        This may involve releasing resources or closing connections.
        Returns: Nothing
        """
        pass


from workout_monitor.tracker.stream.utils import IMX500Receiver

HEADLESS = False
from importlib.metadata import distribution, PackageNotFoundError
import cv2
from workout_monitor.tracker.stream.stream_frame import Frame

MODEL = "ssd_mobilenet_v1"


class IMX500(AbstractSource):
    def __init__(self):
        super().__init__()
        self.width = 300
        self.height = 300
        self.imx500data = IMX500Receiver()
        self.imx_data = None
        self.output_tensor = None
        self.imx_cfg_properties = {}
        self.scale_bbox_to_image = True

    def get_frame(self):
        # 1. Get result
        self.imx_data = self.imx500data.fifo_pop()

        if HEADLESS:
            image = None
        else:
            image = self.image()

        return Frame(
            result=self.imx500_tensor_converter(self.imx_data),
            image_b64="Input",  # Place holder, legacy.
            image=image,
            width=self.width,
            height=self.height,
            FPS=self.FPS,
            timestamp="timestamp",
        )

    def work_put(self, job):
        self.imx500data.annotation_queue_put(job)

    def image(self):
        self.input_tensor_image = IVS.input_tensor_image(
            self.imx_data["Imx500InputTensor"], (self.width, self.height), (256, 256, 256), (0, 0, 0)
        )
        self.input_tensor_image = cv2.UMat(self.input_tensor_image)
        return self.input_tensor_image

    def __iter__(self):
        pass

    def __next__(self):
        return self.get_frame()

    def __enter__(self):
        self.imx_cfg_properties = self.imx500data.start_stream()
        self.imx5000_tensor_info_parser(self.imx_cfg_properties["meta_data"])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO close connection
        pass

    def send(self, topic, message):
        # TODO: Obsolete
        pass

    def imx5000_tensor_info_parser(self, tensor_info):
        network_name, *tensor_data_num, num_tensors = struct.unpack("64s16II", bytes(tensor_info["Imx500OutputTensorInfo"]))
        network_name, self.width, self.height, num_channels = struct.unpack("64sIII", bytes(tensor_info["Imx500InputTensorInfo"]))

        network_name = network_name.decode("utf-8").rstrip("\x00")
        self.tensor_data_num = tensor_data_num[:num_tensors]

        print(f"Network: {network_name} Input tensor dimension: {self.width} x {self.height}.")

    def imx500_tensor_converter(self, imx_data):
        output_tensor = np.array(imx_data["Imx500OutputTensor"])
        output_tensor_split = np.array_split(output_tensor, np.cumsum(self.tensor_data_num[:-1]))

        bbox = output_tensor[0 : self.tensor_data_num[0]].reshape(4, 10).T

        if self.scale_bbox_to_image:
            # This is the case where the imx500 image is used as image:
            new_bbox = np.empty((0, 4))
            for row in bbox:
                new_bbox = np.vstack([new_bbox, self.imx500_bbox_scale(row)])
            bbox = new_bbox
        else:
            # This is the case where the input tensor is used as image:
            # This format might need to change from X,Y,w,y to X,Y,x,y.
            bbox = bbox[:, 0:4] * np.array([self.width[1], self.height[0], self.width[1], self.height[0]])

        if MODEL == "ssd_mobilenet_v1":
            categories, confs = output_tensor_split[1], output_tensor_split[2]
        else:
            categories, confs = output_tensor_split[2], output_tensor_split[1]

        out_tensor_combined = np.column_stack((bbox, confs, categories))
        return out_tensor_combined

    def imx500_bbox_scale(self, coords):
        """Create a Detection object, recording the bounding box, category and confidence."""
        # Scale the box to the output stream dimensions.
        isp_output_size = Size(*self.imx_cfg_properties["camera_configuration"]["main"]["size"])

        sensor_output_size = Size(*self.imx_cfg_properties["camera_configuration"]["raw"]["size"])
        full_sensor_resolution = Rectangle(*self.imx_cfg_properties["camera_properties"]["ScalerCropMaximum"])

        scaler_crop = Rectangle(*self.imx_cfg_properties["meta_data"]["ScalerCrop"])

        obj_scaled = IMX500.convert_inference_coords(
            coords,
            full_sensor_resolution,
            scaler_crop,
            isp_output_size,
            sensor_output_size,
        )
        # Transform to use absolute coordinates instead of relative.
        return np.array([obj_scaled.x, obj_scaled.y, obj_scaled.x + obj_scaled.width, obj_scaled.y + obj_scaled.height])
