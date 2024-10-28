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


def bbox_overlaps(atlbrs, btlbrs):
    """
    Compute the Intersection over Union (IoU) between two sets of bounding boxes.

    :param atlbrs: An array of shape (M, 4) representing the first set of bounding boxes.
    :param btlbrs: An array of shape (N, 4) representing the second set of bounding boxes.
    :return: A matrix of shape (M, N) where element (i, j) is the IoU between the ith bounding box in atlbrs and the jth bounding box in btlbrs.
    """

    # Expand dimensions to allow broadcasting
    a_areas = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    b_areas = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])

    # Broadcast the bounding boxes' coordinates for easier computation
    inter_x1 = np.maximum(atlbrs[:, None, 0], btlbrs[:, 0])
    inter_y1 = np.maximum(atlbrs[:, None, 1], btlbrs[:, 1])
    inter_x2 = np.minimum(atlbrs[:, None, 2], btlbrs[:, 2])
    inter_y2 = np.minimum(atlbrs[:, None, 3], btlbrs[:, 3])

    # Compute the area of intersection rectangle
    inter_area = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    # Compute the area of union
    union_area = (a_areas[:, None] + b_areas) - inter_area

    # Compute the IoU
    ious = inter_area / union_area

    return ious
