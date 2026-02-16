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
from modlib.apps.annotate import Annotator, Color
from typing import Tuple, List, Optional
from enum import Enum, auto

from .menu_functions import *


menu_functions = [DefinePolygon(), DefineCircle(), DefineRectangle(), Save(), SaveAreas(), LoadAreas(), DefineROI(), HideMenu(), Delete()]


class RectangleGrid:
    def __init__(self, menu_functions: List, annotator: Annotator, cols: int = 4, rows: int = 3):
        self.cols = cols
        self.rows = rows
        self.annotator = annotator
        self.menu_functions = menu_functions
        self.rectangles: List[NamedRectangle] = self._generate_grid(menu_functions)

    def _generate_grid(self, menu_functions) -> List[NamedRectangle]:
        rects = []
        gap = 0.05  # Define the gap size

        # Calculate cell width and height considering the gap
        cell_width = (1.0 - (self.cols - 1) * gap) / self.cols
        cell_height = (1.0 - (self.rows - 1) * gap) / self.rows

        for row in range(self.rows):
            for col in range(self.cols):
                x1 = col * (cell_width + gap)
                y1 = row * (cell_height + gap)
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                # Proceed with your drawing logic using (x1, y1, x2, y2)

                if menu_functions:
                    func = menu_functions.pop()
                    rects.append(NamedRectangle(func, (x1, y1), (x2, y2), func.name))
                else:
                    break

        return rects

    def get_rectangle_at(self, point: Tuple[float, float]) -> Optional[NamedRectangle]:
        for rect in self.rectangles:
            if rect.contains(point):
                return rect
        return None

    def annotate_rectangles(self, frame):
        color = (200, 0, 1)

        for rect in self.rectangles:
            frame.image = self.annotator.annotate_area(frame, rect, color, label=rect.name)


def to_relative_coords(x: float, y: float, size: Tuple[int, int]) -> Tuple[float, float]:
    width, height = size
    x, y = x / width, y / height

    x = 1 if x > 1 else x
    y = 1 if y > 1 else y

    return (x, y)


class AppState(Enum):
    INIT = auto()
    NORMAL = auto()
    MENU = auto()
    SELECTED_MENU_FUNC = auto()


class CVMenuStateMachine:
    def __init__(self):
        self.state = AppState.INIT
        self.tick_counter = 0
        self.selected_menu_function = HideMenu()
        self.annotator = Annotator(Color.blue, thickness=2, text_scale=0.5)
        self.rects = RectangleGrid(menu_functions, self.annotator)
        self.states = {
            AppState.INIT: self.state_init,
            AppState.NORMAL: self.state_menu,
            AppState.MENU: self.state_menu,
            AppState.SELECTED_MENU_FUNC: self.state_selected_menu_func,
        }
        self.window_size = 640, 480

    def change_state(self, new_state):
        # print(f"State changed from {self.state.name} to {new_state.name}")
        self.state = new_state

    def tick(self, frame):
        self.tick_counter += 1
        self.states[self.state](frame)

    def state_init(self, frame):
        if self.tick_counter >= 10:
            self.window_size = np.flip(frame.image.shape[0:2], axis=0)
            print(f"w size is {self.window_size}")
            cv2.setMouseCallback("Application", self.capture_event)
            self.change_state(AppState.SELECTED_MENU_FUNC)

    def state_menu(self, frame):
        self.rects.annotate_rectangles(frame)
        if self.selected_menu_function is not None:
            self.change_state(AppState.SELECTED_MENU_FUNC)

    def state_selected_menu_func(self, frame):
        ret = self.selected_menu_function.frame_process(frame)
        if ret:
            self.selected_menu_function = None
            self.change_state(AppState.MENU)

    def capture_event(self, event, x, y, flags, param):
        x, y = to_relative_coords(x, y, self.window_size)
        if self.selected_menu_function is not None:
            self.selected_menu_function.mouse_function(event, x, y, flags, param)
        else:
            if event == cv2.EVENT_LBUTTONUP:
                self.selected_menu_function = self.rects.get_rectangle_at((x, y))
