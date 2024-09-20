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

import struct
import cv2

from abc import ABC, abstractmethod
import numpy as np

from libcamera import Rectangle, Size
from highvis.tracker.stream.utils import IMX500Receiver
from highvis.tracker.stream.stream_frame import Frame


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


class IMX500(AbstractSource):
    def __init__(self, configuration=None):
        super().__init__()
        self.imx500data = IMX500Receiver(configuration)
        self.imx_data = None
        self.width = 1
        self.height = 1

    def get_frame(self):
        self.imx_data = self.imx500data.fifo_pop()

        return Frame(
            result=self.imx_data,
            FPS=self.FPS,
        )

    def work_put(self, job):
        self.imx500data.annotation_queue_put(job)

    def __iter__(self):
        pass

    def __next__(self):
        return self.get_frame()

    def __enter__(self):
        """ " Start the meta data stream."""
        self.imx500data.start_stream()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # TODO close connection
        pass
