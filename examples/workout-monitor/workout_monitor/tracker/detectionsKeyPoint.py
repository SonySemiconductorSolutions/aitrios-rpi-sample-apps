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

class BaseTensor():
    """Base tensor class with additional methods for easy manipulation and device handling."""

    def __init__(self, data, orig_shape) -> None:
        """
        Initialize BaseTensor with prediction data and the original shape of the image.
        """
        assert isinstance(data, (np.ndarray))
        self.data = data
        self.orig_shape = orig_shape
        
    def __len__(self):  # override len(results)
        """Return the length of the underlying data tensor."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return a new BaseTensor instance containing the specified indexed elements of the data tensor."""
        #print(self.data)
        #print(idx)
        return self.__class__(self.data[idx], self.orig_shape)

class Results():

    def __init__(
        self, orig_img, boxes=None,probs=None, keypoints=None
    ) -> None:
        """
        Initialize the Results class for storing and manipulating inference results.
        """
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None  # native size boxes
        self.probs = Probs(probs) if probs is not None else None
        self.keypoints = Keypoints(keypoints, self.orig_shape) if keypoints is not None else None
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"
        
    def __getitem__(self, idx):
        """Return a Results object for a specific index of inference results."""
        return self._apply("__getitem__", idx)

    def __len__(self):
        """Return the number of detections in the Results object from a non-empty attribute set (boxes, masks, etc.)."""
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                return len(v)
                
    def _apply(self, fn, *args, **kwargs):
        """
        Applies a function to all non-empty attributes and returns a new Results object with modified attributes. This
        function is internally called by methods like .to(), .cuda(), .cpu(), etc.
        """
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r
        
    def cpu(self):
        """Returns a copy of the Results object with all its tensors moved to CPU memory."""
        return self._apply("cpu")

    def numpy(self):
        """Returns a copy of the Results object with all tensors as numpy arrays."""
        return self._apply("numpy")

    def cuda(self):
        """Moves all tensors in the Results object to GPU memory."""
        return self._apply("cuda")

    def to(self, *args, **kwargs):
        """Moves all tensors in the Results object to the specified device and dtype."""
        return self._apply("to", *args, **kwargs)

    def new(self):
        """Returns a new Results object with the same image, path, names, and speed attributes."""
        return Results(orig_img=self.orig_img, speed=self.speed)

    def show(self, *args, **kwargs):
        """Show the image with annotated inference results."""
        self.plot(show=True, *args, **kwargs)

    def save(self, filename=None, *args, **kwargs):
        """Save annotated inference results image to file."""
        if not filename:
            filename = f"results_{Path(self.path).name}"
        self.plot(save=True, filename=filename, *args, **kwargs)
        return filename

    def verbose(self):
        """Returns a log string for each task in the results, detailing detection and classification outcomes."""
        log_string = ""
        probs = self.probs
        boxes = self.boxes
        if len(self) == 0:
            return log_string if probs is not None else f"{log_string}(no detections), "
        if probs is not None:
            log_string += f"{', '.join(f'{self.names[j]} {probs.data[j]:.2f}' for j in probs.top5)}, "
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()  # detections per class
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string


class Boxes(BaseTensor):

    def __init__(self, boxes, orig_shape) -> None:
        """
        Initialize the Boxes class with detection box data and the original image shape.
        """
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7}, f"expected 6 or 7 values but got {n}"  # xyxy, track_id, conf, cls
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape
        
class Keypoints(BaseTensor):
   
    def __init__(self, keypoints, orig_shape) -> None:
        """Initializes the Keypoints object with detection keypoints and original image dimensions."""
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
        if keypoints.shape[2] == 3:  # x, y, conf
            mask = keypoints[..., 2] < 0.5  # points with conf < 0.5 (not visible)
            keypoints[..., :2][mask] = 0
        super().__init__(keypoints, orig_shape)
        self.has_visible = self.data.shape[-1] == 3
