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

import json
import argparse
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from picamera2 import Picamera2, MappedArray, CompletedRequest


class parkingPtsSelection:
    def __init__(self):
        """Initializes the UI for selecting parking zone points in a tkinter window."""
        self.args = self.get_args()

        if self.args.filename[-5:] != ".json":  # Check to see if filename ends in .json to create json file properly
            raise Exception("Filename must end in .json")

        self.picam2 = Picamera2()
        self.camera_modes = self.picam2.sensor_modes
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": 30, "ScalerCrop": self.picam2.sensor_modes[0]["crop_limits"]}, buffer_count=28
        )
        self.picam2.start(config, show_preview=True)

        self.picam2.pre_callback = self.run
        import tkinter as tk

        self.tk = tk
        self.master = tk.Tk()
        self.master.title("parking Zones Points Selector")

        # Disable window resizing
        self.master.resizable(False, False)

        # Setup canvas for image display
        self.canvas = self.tk.Canvas(self.master, bg="white")

        # Setup buttons
        button_frame = self.tk.Frame(self.master)
        button_frame.pack(side=self.tk.TOP)

        self.tk.Button(button_frame, text="Take Image", command=self.take_image).grid(row=0, column=0)
        self.tk.Button(button_frame, text="Remove Last BBox", command=self.remove_last_bounding_box).grid(
            row=0, column=1
        )
        self.tk.Button(button_frame, text="Save", command=self.save_to_json).grid(row=0, column=2)
        self.tk.Button(button_frame, text="Reset crop", command=self.reset_crop).grid(row=0, column=3)
        # Initialize properties
        self.image_path = None
        self.image = None
        self.canvas_image = None
        self.bounding_boxes = []
        self.current_box = []
        self.img_width = 0
        self.img_height = 0

        self.crop_points = []
        self.crop_box = 0
        self.cropped = False

        # Constants
        self.canvas_max_width = 1280
        self.canvas_max_height = 720

        self.master.mainloop()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--filename", type=str, required=False, default="config.json", help="Path of the json file")
        return parser.parse_args()

    def run(self, request: CompletedRequest):
        self.request = request

    def reset_crop(self):

        self.crop_box = self.picam2.sensor_modes[0]["crop_limits"]
        print(self.crop_box)
        self.picam2.set_controls({"ScalerCrop": self.picam2.sensor_modes[0]["crop_limits"]})
        self.cropped = False

    def take_image(self):
        """Upload an image and resize it to fit canvas."""
        with MappedArray(self.request, stream="main") as m:

            self.image = Image.fromarray(m.array)
            self.img_width, self.img_height = self.image.size

            # Calculate the aspect ratio and resize image
            aspect_ratio = self.img_width / self.img_height
            if aspect_ratio > 1:
                # Landscape orientation
                canvas_width = min(self.canvas_max_width, self.img_width)
                canvas_height = int(canvas_width / aspect_ratio)
            else:
                # Portrait orientation
                canvas_height = min(self.canvas_max_height, self.img_height)
                canvas_width = int(canvas_height * aspect_ratio)

            # Check if canvas is already initialized
            if self.canvas:
                self.canvas.destroy()  # Destroy previous canvas

            self.canvas = self.tk.Canvas(self.master, bg="white", width=canvas_width, height=canvas_height)
            resized_image = self.image.resize((canvas_width, canvas_height), Image.LANCZOS)
            self.canvas_image = ImageTk.PhotoImage(resized_image)
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)

            self.canvas.pack(side=self.tk.BOTTOM)
            self.canvas.bind("<Button-1>", self.on_canvas_click)
            self.canvas.bind("<Button-3>", self.on_canvas_click_right)

            # Reset bounding boxes and current box
            self.bounding_boxes = []
            self.current_box = []

    def on_canvas_click_right(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        if self.cropped:
            print("Reset crop before setting new.")
            return
        x = (event.x / self.img_width) * self.camera_modes[0]["crop_limits"][2]
        y = (event.y / self.img_height) * self.camera_modes[0]["crop_limits"][3]

        self.crop_points.append((int(x), int(y)))

        x0, y0 = event.x - 3, event.y - 3
        x1, y1 = event.x + 3, event.y + 3

        if len(self.crop_points) == 2:
            width = int(self.crop_points[1][0] - self.crop_points[0][0])
            height = int(self.crop_points[1][1] - self.crop_points[0][1])
            if width <= 0 or height <= 0:
                self.crop_points.pop()
                print(f"Height or width is zero or less: w:{width} height:{height}.")
                return
            self.crop_box = (self.crop_points[0][0], self.crop_points[0][1], width, height)
            self.picam2.sensor_modes[0]["crop_limits"]
            self.picam2.set_controls({"ScalerCrop": self.crop_box})
            self.crop_points = []
            self.cropped = True

            self.take_image()
        self.canvas.create_oval(x0, y0, x1, y1, fill="green")

    def on_canvas_click(self, event):
        """Handle mouse clicks on canvas to create points for bounding boxes."""
        self.current_box.append((event.x, event.y))
        x0, y0 = event.x - 3, event.y - 3
        x1, y1 = event.x + 3, event.y + 3
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")

        if len(self.current_box) == 4:
            self.bounding_boxes.append(self.current_box)
            self.draw_bounding_box(self.current_box)
            print(self.current_box, self.bounding_boxes)
            self.current_box = []

    def draw_bounding_box(self, box):
        """
        Draw bounding box on canvas.

        Args:
                box (list): Bounding box data
        """
        for i in range(4):
            x1, y1 = box[i]
            x2, y2 = box[(i + 1) % 4]
            self.canvas.create_line(x1, y1, x2, y2, fill="blue", width=2)

    def remove_last_bounding_box(self):
        """Remove the last drawn bounding box from canvas."""
        if self.bounding_boxes:
            self.bounding_boxes.pop()  # Remove the last bounding box
            self.canvas.delete("all")  # Clear the canvas
            self.canvas.create_image(0, 0, anchor=self.tk.NW, image=self.canvas_image)  # Redraw the image

            # Redraw all bounding boxes
            for box in self.bounding_boxes:
                self.draw_bounding_box(box)

    def save_to_json(self):
        """Saves rescaled bounding boxes to 'bounding_boxes.json' based on image-to-canvas size ratio."""
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        width_scaling_factor = self.img_width / canvas_width
        height_scaling_factor = self.img_height / canvas_height
        configuration = {}
        bounding_boxes_data = []
        for box in self.bounding_boxes:
            rescaled_box = []
            for x, y in box:
                rescaled_x = int(x * width_scaling_factor)
                rescaled_y = int(y * height_scaling_factor)
                rescaled_box.append((rescaled_x, rescaled_y))
            bounding_boxes_data.append({"points": rescaled_box})
        configuration["areas"] = bounding_boxes_data
        configuration["crop_settings"] = {
            "x": self.crop_box[0],
            "y": self.crop_box[1],
            "w": self.crop_box[2],
            "h": self.crop_box[3],
        }
        with open(self.args.filename, "w") as json_file:
            json.dump(configuration, json_file, indent=1)
        print(f"Saved to {self.args.filename}.")

        # messagebox.showinfo("Success", "Bounding boxes saved to bounding_boxes.json")


def start_pts_demo():
    cv2.startWindowThread()
    ptsSelector = parkingPtsSelection()
    ptsSelector.run


if __name__ == "__main__":
    start_pts_demo()
    exit()
