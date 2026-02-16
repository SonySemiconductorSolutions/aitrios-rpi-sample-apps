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

import json
from collections import deque
from modlib.apps.area import Rectangle, Circle, Area
from modlib.devices.frame import ROI


def Rect2ROI(rectangle: Rectangle) -> ROI:
    """
    Convert a rectangle from (top_left, bottom_right) format to (top_left, width, height).

    Parameters:
    - top_left (tuple): Coordinates of the top-left corner (x, y).
    - bottom_right (tuple): Coordinates of the bottom-right corner (x, y).

    Returns:
    - tuple: An ROI object with (left, top, width, height)
    """
    x1, y1 = rectangle.top_left
    x2, y2 = rectangle.bottom_right
    width = x2 - x1
    height = y2 - y1
    left, top = rectangle.top_left
    roi = ROI(left, top, width, height)
    return roi


class ConfigReader:
    def __init__(self, path="areas.json"):
        self.path = path
        self.areas = deque(maxlen=20)
        self.roi_rectangle = deque(maxlen=1)

    def get_roi(self) -> ROI:
        self.get()
        if len(self.roi_rectangle) >= 1:
            roi = Rect2ROI(self.roi_rectangle[0])
            return roi
        return ROI(0, 0, 1, 1)

    def get_areas(self) -> deque:
        return self.areas

    def get(self):
        try:
            with open(self.path, "r") as file:
                data = json.load(file)
                print("File loaded successfully.")
        except FileNotFoundError:
            print(f"The file at {self.path} does not exist.")
            return True
        except json.JSONDecodeError:
            print(f"The file at {self.path} is not a valid JSON file.")
            return True
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return True

        if "shapes" in data:
            self.areas.clear()
            for shape in data["shapes"]:
                if shape["type"] == "Rectangle":
                    self.areas.append(Rectangle.from_dict(shape))
                elif shape["type"] == "Circle":
                    self.areas.append(Circle.from_dict(shape))
                elif shape["type"] == "Area":
                    self.areas.append(Area.from_dict(shape))
        if "roi" in data:
            self.roi_rectangle.clear()
            self.roi_rectangle.append(Rectangle.from_dict(data["roi"]))

        return True

    def save(self):
        shapes = [shape.to_dict() for shape in self.areas]
        result = {"shapes": shapes}
        if len(self.roi_rectangle) >= 1:
            result["roi"] = self.roi_rectangle[0].to_dict()
        print(f"Saving to {self.path}.")
        with open(self.path, "w") as file:
            json.dump(result, file, indent=4)
