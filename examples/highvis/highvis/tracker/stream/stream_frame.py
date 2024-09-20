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


class Frame(object):
    def __init__(self, result=None, image_b64=None, image=None, width=None, height=None, FPS=0, timestamp=None) -> None:
        self.result = result
        self.image_b64 = image_b64
        self.image = image
        self.width = width
        self.height = height
        self.fps = FPS
        self.timestamp = timestamp

    def display(self, scale_factor=1):
        self.annotate_fps(self.image, self.fps)

        if scale_factor > 1:
            self.image = cv2.resize(self.image, (int(self.width * scale_factor), int(self.height * scale_factor)))

        cv2.imshow("Application", self.image)
        key = cv2.waitKey(1) & 0xFF
        # 'ESC' key or Window is closed manually
        if key == 27 or cv2.getWindowProperty("Application Esc to close", cv2.WND_PROP_VISIBLE) < 1:
            raise StopIteration()

    @staticmethod
    def annotate_fps(frame, fps):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
