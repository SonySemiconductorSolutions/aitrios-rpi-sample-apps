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

import numpy as np
from typing import List, Tuple, Iterator, Union


class ToTrack:
    output_results: np.ndarray
    img_info: Tuple[int, int]
    img_size: Tuple[int, int]


class Detections:
    """
    Attributes:
    - bbox (np.ndarray):            Array of shape (n, 4) the bounding boxes [x1, y1, x2, y2] of N detections
    - confidence (np.ndarray):      Array of shape (n,) the confidence of N detections
    - class_id (np.ndarray):        Array of shape (n,) the class id of N detections
    - tracker_id (np.ndarray):      Array of shape (n,) the tracker id of N detections
    - frame_size (Tuple(int, int)): Tuple of (height, width) the size of the frame

    Properties:
    - area (np.ndarray):            Array of shape (n,) the area of the bounding boxes of N detections
    - bbox_width (np.ndarray):      Array of shape (n,) the width of the bounding boxes of N detections
    - bbox_height (np.ndarray):     Array of shape (n,) the height of the bounding boxes of N detections
    """

    bbox: np.ndarray
    confidence: np.ndarray
    class_id: np.ndarray
    tracker_id: np.ndarray
    frame_size: Tuple[int, int]

    def __init__(self, frame_height: int, frame_width: int) -> None:
        self.frame_size = (frame_height, frame_width)

        self.bbox = np.empty((0, 4))
        self.confidence = np.empty((0,))
        self.class_id = np.empty((0,))
        self.tracker_id = np.empty((0,))

    ## OPERATORS
    def __len__(self):
        """
        Return the number of detections.
        """
        return len(self.bbox)

    def __copy__(self):
        """
        Return a copy of the current detections.
        """
        new_instance = Detections(*self.frame_size)
        new_instance.bbox = np.copy(self.bbox)
        new_instance.confidence = np.copy(self.confidence)
        new_instance.class_id = np.copy(self.class_id)
        new_instance.tracker_id = np.copy(self.tracker_id)

        return new_instance

    def copy(self):
        return self.__copy__()

    def __getitem__(self, index: Union[int, slice, List[int], np.ndarray]) -> "Detections":
        """
        Returns a new Detections object with the selected detections.
        Could be a subsection of the current detections.
        """
        if isinstance(index, int):
            index = [index]

        res = self.copy()
        res.bbox = self.bbox[index]
        res.confidence = self.confidence[index]
        res.class_id = self.class_id[index]
        res.tracker_id = self.tracker_id[index]
        return res

    def __iter__(self) -> Iterator[Tuple[np.ndarray, float, int, int]]:
        for i in range(len(self)):
            yield (
                self.bbox[i],
                self.confidence[i],
                self.class_id[i],
                self.tracker_id[i],
            )

    def __add__(self, other: "Detections") -> "Detections":
        if not isinstance(other, Detections):
            raise TypeError(f"Unsupported operand type(s) for +: 'Detections' and '{type(other)}'")

        result = self.copy()
        result.bbox = np.vstack((result.bbox, other.bbox))
        result.confidence = np.concatenate([self.confidence, other.confidence])
        result.class_id = np.concatenate([self.class_id, other.class_id])
        result.tracker_id = np.concatenate([self.tracker_id, other.tracker_id])
        return result

    # __eq__
    # ... / ne

    ## LOAD SOURCES
    def clear(self):
        """
        Clear the current detections.
        """
        self.bbox = np.empty((0, 4))
        self.confidence = np.empty((0,))
        self.class_id = np.empty((0,))
        self.tracker_id = np.empty((0,))

    def update(self, result):
        # BUG: check result type as python doesn't allow for function overloading

        # SSD IMX
        if self._is_SSD_IMX(result):
            self._load_SSD_IMX(result)

        # SSD Mobile Net
        elif self._is_ssd_mobilenet_format(result):
            self._load_SSD(result)

        # C8Y
        elif self._is_c8y_format(result):
            self._load_c8y(result)

        else:
            raise ValueError(f"Unsupported input format for detections update method: {type(result)} \n Got: {result}")

    def _is_ssd_mobilenet_format(self, result):
        if not isinstance(result, (list, tuple, np.ndarray)):
            return False

        if np.shape(result) != (1, 1, 100, 7):
            return False

        # Check if the values in the batch index are consistent with batch size of 1
        if not all(result[0, 0, :, 0] == 0):
            return False

        # Check if the confidence scores are between 0 and 1
        confidence_scores = result[0, 0, :, 2]
        if not all(0 <= score <= 1 for score in confidence_scores):
            return False

        return True

    def _load_SSD(self, result):
        confidence = result[0, 0, :, 2]
        class_id = result[0, 0, :, 1].astype(int)
        bbox = result[0, 0, :, 3:7]

        # Filter out zero detections
        non_zero_detections = np.any(bbox != 0, axis=1)

        self.confidence = confidence[non_zero_detections]
        self.class_id = class_id[non_zero_detections]
        self.bbox = bbox[non_zero_detections] * np.array([self.frame_size[1], self.frame_size[0], self.frame_size[1], self.frame_size[0]])
        self.tracker_id = np.full(self.class_id.shape, -1)

    def _is_c8y_format(self, result):
        """
        A dictionary representing a frame of detected objects.
        (e.g. one frame dictionary)
            {
                "T": "2023-09-21T09:09:18.704475+00:00",
                "O": [
                    {
                        "C": 15,
                        "P": 0.993,
                        "x": 639,
                        "y": 239,
                        "X": 687,
                        "Y": 325
                    },
                    ...
                ]
            }
        """

        # Ensure it's a dictionary with the keys 'T' and 'O' &
        if (not isinstance(result, dict)) or ("T" not in result) or ("O" not in result):
            return False
        if not isinstance(result["O"], list):
            return False

        # Each detected object in 'O' should be a dictionary with keys 'C', 'P', 'x', 'y', 'X', and 'Y'.
        for obj in result["O"]:
            if not isinstance(obj, dict) or not all(key in obj for key in ["C", "P", "x", "y", "X", "Y"]):
                return False

        return True

    def _load_c8y(self, result):
        # result is json string
        for obj in result["O"]:
            c, p, x1, y1, x2, y2 = obj["C"], obj["P"], obj["x"], obj["y"], obj["X"], obj["Y"]

            self.bbox = np.vstack((self.bbox, np.array([x1, y1, x2, y2])))
            self.confidence = np.append(self.confidence, c)
            self.class_id = np.append(self.class_id, p)

        self.tracker_id = np.full(self.class_id.shape, -1)

    def _is_SSD_IMX(self, result):
        # Check if result is a numpy array
        if not isinstance(result, np.ndarray):
            print("Result of wrong type.")
            return False

        # Check if result has the correct shape (N, 6)
        # Shape (10,6) for SSD_Mobilenet and (300,6) for NanoDet.
        if result.shape != (10, 6) and result.shape != (300, 6):
            print("Tensor has wrong shape: ", result.shape)
            return False

        # Check if the 5th column contains values between 0 and 1 (confidence scores)
        if not ((0 <= result[:, 4]) & (result[:, 4] <= 1)).all():
            print("Incorrect confidence score.")
            return False

        # Check if the 6th column contains integer values (class IDs)
        if not (result[:, 5] == result[:, 5].astype(int)).all():
            print("Class columt of incorrect type.")
            return False

        return True

    def _load_SSD_IMX(self, result):
        mask_array = result[:, 4] > 0.0
        # bbox = result[:, 0:4] * np.array([self.frame_size[1], self.frame_size[0], self.frame_size[1], self.frame_size[0]])
        bbox = result[:, 0:4]

        self.bbox = bbox[mask_array]
        self.confidence = result[mask_array, 4]
        self.class_id = result[mask_array, 5].astype(int)
        self.tracker_id = np.full(self.class_id.shape, -1)

        # TODO: Dynamic update frame size
        # something like
        # self.frame_size = (result["widht"], result["height"])

    ### OTHER
    # Other metods to allow easy connection to other parts of the library
    # (e.g. input format of the tracker)
    def to_tracker(self) -> ToTrack:

        out = ToTrack()
        out.img_info = self.frame_size
        out.img_size = self.frame_size
        out.output_results = np.hstack((self.bbox, self.confidence[:, np.newaxis]))

        return out

    ## PROPERTIES
    @property
    def area(self) -> np.ndarray:
        widths = self.bbox[:, 2] - self.bbox[:, 0]
        heights = self.bbox[:, 3] - self.bbox[:, 1]
        return widths * heights

    @property
    def bbox_width(self) -> np.ndarray:
        return self.bbox[:, 2] - self.bbox[:, 0]

    @property
    def bbox_height(self) -> np.ndarray:
        return self.bbox[:, 3] - self.bbox[:, 1]
