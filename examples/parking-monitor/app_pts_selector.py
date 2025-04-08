#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
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
import argparse
import cv2
from PIL import Image, ImageTk
from picamera2 import Picamera2, MappedArray, CompletedRequest
import tkinter


class QueuePtsSelection:
    def __init__(self):
        """Start up UI for queue area points in a tkinter window."""
        self.args = self.get_args()

        self.image_path = None
        self.image = None
        self.focus_image = None
        self.bboxes = []
        self.box = []
        self.img_width = 0
        self.img_height = 0

        self.canvas_max_width = 1280
        self.canvas_max_height = 720

        if (
            self.args.filename[-5:] != ".json"
        ):  # Check to see if filename ends in .json to create json file properly
            raise Exception("Filename must end in .json")

        # Picamera2 setup
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": 30}, buffer_count=28
        )
        self.picam2.start(config, show_preview=False)
        self.picam2.pre_callback = self.run

        self.Tkinter = tkinter
        self.main = tkinter.Tk()
        self.main.title("Queue Zones Points Selector")

        self.main.resizable(False, False)

        # Image Display
        self.canvas = self.Tkinter.Canvas(self.main, bg="white")

        # Buttons
        button_canvas = self.Tkinter.Frame(self.main)
        button_canvas.pack(side=self.Tkinter.TOP)

        self.Tkinter.Button(
            button_canvas, text="Take Image", command=self.take_image
        ).grid(row=0, column=0)
        self.Tkinter.Button(
            button_canvas,
            text="Remove Previous BBox",
            command=self.remove_previous_bbox,
        ).grid(row=0, column=1)
        self.Tkinter.Button(button_canvas, text="Save", command=self.export_json).grid(
            row=0, column=2
        )

        self.main.mainloop()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--filename", type=str, required=True, help="Path of the json file"
        )
        return parser.parse_args()

    def run(self, request: CompletedRequest):
        self.request = request

    def take_image(self):
        with MappedArray(self.request, stream="main") as m:
            self.image = Image.fromarray(m.array)
            self.img_width, self.img_height = self.image.size

            aspect_ratio = self.img_width / self.img_height
            canvas_width = min(self.canvas_max_width, self.img_width)
            canvas_height = int(canvas_width / aspect_ratio)

            if self.canvas:
                self.canvas.destroy()

            self.canvas = self.Tkinter.Canvas(
                self.main, bg="white", width=canvas_width, height=canvas_height
            )
            resized_image = self.image.resize(
                (canvas_width, canvas_height), Image.LANCZOS
            )
            self.main_image = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(
                0, 0, anchor=self.Tkinter.NW, image=self.main_image
            )

            self.canvas.pack(side=self.Tkinter.BOTTOM)
            self.canvas.bind(
                "<Button-1>", self.detect_canvas_click_left
            )  # Checking for mouse left click event

            # Redeclare
            self.bboxes = []
            self.box = []

    def detect_canvas_click_left(self, click):
        self.box.append((click.x, click.y))
        x0, y0 = click.x - 3, click.y - 3
        x1, y1 = click.x + 3, click.y + 3
        self.canvas.create_oval(x0, y0, x1, y1, fill="blue")

        if len(self.box) == 4:
            self.bboxes.append(self.box)
            self.draw_bbox(self.box)
            self.box = []

    def draw_bbox(self, box):
        for i in range(4):
            x1, y1 = box[i]
            x2, y2 = box[(i + 1) % 4]
            self.canvas.create_line(x1, y1, x2, y2, fill="orange", width=2)

    def remove_previous_bbox(self):
        if self.bboxes:
            self.bboxes.pop()
            self.canvas.delete("all")
            self.canvas.create_image(
                0, 0, anchor=self.Tkinter.NW, image=self.main_image
            )
            # Redraw bboxes
            for box in self.bboxes:
                self.draw_bbox(box)

    def normalize(self, box):
        norm_box = []
        for point in box:
            Npoint = (point[0] / 640, point[1] / 480)
            norm_box.append(Npoint)
        return norm_box

    def export_json(self):
        """Saves rescaled bounding boxes to json file"""
        bboxes_data = []
        print(self.bboxes)
        for box in self.bboxes:
            box = self.normalize(box)
            bboxes_data.append({"points": box})
        print(bboxes_data)
        with open(self.args.filename, "w") as json_file:
            json.dump(bboxes_data, json_file, indent=4)


def start_pts_demo():
    cv2.startWindowThread()
    QueuePtsSelection()


if __name__ == "__main__":
    start_pts_demo()
    exit()
