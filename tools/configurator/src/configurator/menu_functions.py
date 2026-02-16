#
# Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import numpy as np

from modlib.apps.area import Rectangle, Circle, Area
from modlib.apps.annotate import Annotator, Color

from typing import Tuple
from .config_reader import ConfigReader

config = ConfigReader()

class NamedRectangle(Rectangle):
    def __init__(self, function, top_left: Tuple[float, float], bottom_right: Tuple[float, float], name: str):
        self.name = name
        self.function = function
        super().__init__(top_left, bottom_right)

    def contains(self, point: Tuple[float, float]) -> bool:
        x, y = point
        x1, y1 = self.top_left
        x2, y2 = self.bottom_right
        return x1 <= x <= x2 and y1 <= y <= y2

    def mouse_function(self, event, x, y, flags, param):
        self.function.mouse_function(event, x, y, flags, param)

    def frame_process(self, frame):
        return self.function.frame_process(frame)


class MenuItem:
    def __init__(self):
        self.name = "MenuItem"

    def mouse_function(self, event, x, y, flags, param):
        self.finished = True

    def frame_process(self, frame):
        return True

    def on_select(self):
        # init variables on new selection.
        pass


class HideMenu(MenuItem):
    def __init__(self):
        self.name = "CloseMenu"
        self.finished = False

    def mouse_function(self, event, x, y, flags, param):
        if event == cv2.EVENT_MBUTTONDOWN:
            self.finished = True

    def frame_process(self, frame):
        if self.finished:
            self.finished = False
            return True
        return False


class SaveAreas(MenuItem):
    def __init__(self):
        self.name = "Save Area"
        self.finished = False
        self.counter = 0

    def mouse_function(self, event, x, y, flags, param):
        self.finished = True

    def frame_process(self, frame):
        config.save()
        return True


class LoadAreas(MenuItem):
    def __init__(self):
        self.name = "Load Area"
        self.finished = False
        self.counter = 0

    def mouse_function(self, event, x, y, flags, param):
        self.finished = True

    def frame_process(self, frame):
        config.get()
        return True


class Save:
    def __init__(self):
        self.name = "Save Image"
        self.finished = False
        self.counter = 0
        self.frame_ticks = 0

    def mouse_function(self, event, x, y, flags, param):
        self.finished = True

    def frame_process(self, frame):
        if self.frame_ticks >= 2:
            fname = f"./saved_{self.counter:03d}.jpg"
            self.counter += 1
            cv2.imwrite(fname, frame.image)
            print(f"Saved image: {fname}")
            self.frame_ticks = 0
            return True
        else:
            self.frame_ticks += 1
        return False


class DefineShape:
    def __init__(self, cls_shape):
        self.shape = cls_shape

        self.finished = False
        self.top_left: Tuple = (0, 0)
        self.bottom_right: Tuple = (1, 1)
        self.moving_area = None
        self.annotator = Annotator(Color.blue, thickness=2, text_scale=0.5)

    def add_top_left(self, x, y):
        if x < self.bottom_right[0] and y < self.bottom_right[1]:
            self.top_left = (x, y)
            self._add_area()
        else:
            print(f"Invalid top left: {x, y}.")

    def add_bottom_right(self, x, y):
        if x > self.top_left[0] and y > self.top_left[1]:
            self.bottom_right = (x, y)
            self._add_area()
        else:
            print(f"Invalid bottom right: {x, y}.")

    def moving(self, x, y):
        if x > self.top_left[0] and y > self.top_left[1]:
            self.moving_area = self.shape(self.top_left, (x, y))
        else:
            print(f"Invalid bottom right: {x, y}. top left {self.top_left}")

    def _add_area(self):
        if self.bottom_right == (1, 1):
            return
        elif self.top_left == (0, 0):
            return
        else:
            config.areas.append(self.shape(self.top_left, self.bottom_right))

            self.moving_area = None
            self.top_left = (0, 0)
            self.bottom_right = (1, 1)

    def mouse_function(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_top_left(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            self.moving(x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.finished = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.add_bottom_right(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            if config.areas:
                config.areas.pop()

    def frame_process(self, frame):
        for area in config.areas:
            frame.image = self.annotator.annotate_area(frame, area, (100, 200, 200))

        if self.moving_area is not None:
            frame.image = self.annotator.annotate_area(frame, self.moving_area, (10, 10, 255))

        if self.finished:
            self.finished = False
            return True
        return False


class DefineROI(DefineShape):
    def __init__(self):
        self.name = "ROI"
        self.moving_point = (0,0)

        super().__init__(Rectangle)

    def moving(self, x, y):
        if x > self.top_left[0] and y > self.top_left[1]:
            self.moving_point = (x,y)

    def _add_area(self):
        if self.bottom_right == (1, 1):
            return
        elif self.top_left == (0, 0):
            return
        else:
            config.roi_rectangle.append(Rectangle(self.top_left, self.bottom_right))
            self.moving_point = (0, 0)
            self.top_left = (0, 0)
            self.bottom_right = (1, 1)

    def frame_process(self, frame):
        if len(config.roi_rectangle) >= 1:
            r = config.roi_rectangle[0]
            frame.image = self.annotator.annotate_area(
                frame, config.roi_rectangle[0], (0, 200, 100), label=f"({r.top_left[0]:1.1f},{r.top_left[1]:1.1f})x({r.bottom_right[0]:1.1f},{r.bottom_right[1]:1.1f})"
            )
        if self.moving_point != (0, 0):
            frame.image = self.annotator.annotate_area(
                frame,
                Rectangle(self.top_left, self.moving_point),
                (255, 1, 1),
                label=f"({self.top_left[0]:1.1f},{self.top_left[1]:1.1f})x({self.moving_point[0]:1.1f},{self.moving_point[1]:1.1f})",
            )
        if self.finished:
            self.finished = False
            return True
        return False


class DefineRectangle(DefineShape):
    def __init__(self):
        self.name = "DrawRect"
        super().__init__(Rectangle)


class DefineCircle(DefineShape):
    def __init__(self):
        self.name = "DrawCirc"
        self.aspect_ratio = None
        self.moving_area = None
        super().__init__(Circle)

    def moving(self, x, y):
        if x > self.top_left[0] and y > self.top_left[1]:
            self.moving_area = Circle(self.top_left, (x, y), aspect_ratio=self.aspect_ratio)

    def _add_area(self):
        if self.bottom_right == (1, 1):
            return
        elif self.top_left == (0, 0):
            return
        else:
            config.areas.append(Circle(self.top_left, point_circumference=self.bottom_right, aspect_ratio=self.aspect_ratio))
            self.moving_area = None
            self.top_left = (0, 0)
            self.bottom_right = (1, 1)

    def frame_process(self, frame):
        if self.aspect_ratio is None:
            w, h = np.flip(frame.image.shape[0:2], axis=0)
            self.aspect_ratio = w / h
        return super().frame_process(frame)


class Delete(MenuItem):
    def __init__(self):
        self.name = "DeleteShape"
        self.finished = False

    def _find_index(self, x, y):
        to_be_removed_index = []
        for i, area in enumerate(config.areas):
            result = cv2.pointPolygonTest(area.points, (x, y), False)
            if result > 0:
                to_be_removed_index.append(i)
        config.areas = [item for i, item in enumerate(config.areas) if i not in to_be_removed_index]

    def mouse_function(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._find_index(x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.finished = True

    def frame_process(self, frame):
        if self.finished:
            self.finished = False
            return True
        return False


class DefinePolygon(DefineShape):
    def __init__(self):
        self.name = "DrawPolygon"
        self.points = []
        self.finished = False
        self.moving_area = None
        super().__init__(Rectangle)

    def moving(self, x, y):
        valid, points = self._test_if_valid(x, y)
        if valid == True and len(points) >= 3:
            tmp_points = np.array(points, np.float32)
            area = Area(tmp_points)
            if area is not None:
                self.moving_area = area

    def _add_area(self):
        if len(self.points) < 3:
            return False
        tmp_points = np.array(self.points, np.float32)
        area = Area(tmp_points)

        if area is not None:
            config.areas.append(area)
            self.points.clear()
            self.moving_area = None
            return True
        return False

    def _add_polygon_point(self, x, y):
        if self._test_if_valid(x, y)[0]:
            self.points.append((x, y))

    def _test_if_valid(self, x, y):
        if x == 0 and y == 0:
            return False, None

        if not (0 <= x <= 1 and 0 <= y <= 1):
            return False, None
        tmp_points = [(x, y) for x, y in self.points]
        tmp_points.append((x, y))

        if len(tmp_points) >= 3:
            tmp_points = np.array(tmp_points, np.float32)
            if not cv2.isContourConvex(tmp_points):
                return False, None
        return True, tmp_points

    def mouse_function(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._add_polygon_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and len(self.points) > 0:
            self.moving(x, y)
        elif event == cv2.EVENT_MBUTTONDOWN:
            ret = self._add_area()
            if ret == False:
                self.finished = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 0:
                self.points.pop()
