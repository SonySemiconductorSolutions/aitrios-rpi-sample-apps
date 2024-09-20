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

import cv2
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)
PURPLE = (128, 0, 128)
BROWN = (139, 69, 19)


class Annotator:
    def __init__(self, font_scale=0.6, font_thickness=1):
        self.colors = [CYAN, CYAN, CYAN, YELLOW, YELLOW]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = font_scale
        self.thickness = font_thickness
        self.line_height = 30
        self.horizontal_offset = 10

    def annotate(self, image, texts):
        for i, text in enumerate(texts):
            position = (self.horizontal_offset, (1 + i) * self.line_height)
            color = self.colors[i % len(self.colors)]
            cv2.putText(image, text, position, self.font, self.font_scale, color, self.thickness)
        return image
