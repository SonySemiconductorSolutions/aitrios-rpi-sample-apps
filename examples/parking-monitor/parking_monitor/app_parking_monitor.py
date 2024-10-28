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
import numpy as np
import signal

from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, MappedArray, CompletedRequest
from picamera2.devices.imx500.postprocess import scale_boxes, scale_coords

from parking_monitor.utils import Detections, shutdown_handler
from parking_monitor import WHITE, RED, GREEN, BLUE
from datetime import datetime

last_boxes = None
last_scores = None
last_keypoints = None

FONT = cv2.FONT_HERSHEY_DUPLEX


class ParkingResult:
    def __init__(self, n_spaces: int):
        self.n_spaces = n_spaces
        self.space = {i: {"status": False, "vehicle_type": -1, "track_id": -1} for i in range(self.n_spaces)}
        self.changed_flag = True
        
        print(str(self))

    def set(self, space_index, class_label: str, track_id: int):
        if space_index <= self.n_spaces:
            if not self.space[space_index]["status"] == True or self.space[space_index]["track_id"] == -1:
                self.changed_flag = True
                iso_timestamp = datetime.now().isoformat()
                self.space[space_index] = {"status": True, "vehicle_type": class_label, "track_id": int(track_id), "timestamp": iso_timestamp}

    def reset(self, space_index):
        if space_index <= self.n_spaces:
            if not self.space[space_index]["status"] == False:
                self.changed_flag = True
                iso_timestamp = datetime.now().isoformat()
                self.space[space_index] = {"status": False, "vehicle_type": -1, "track_id": int(-1), "timestamp": iso_timestamp}


    def __str__(self):
        ret = f"Parking status: {json.dumps(self.space, indent=2)}"
        return ret

    def changed(self):
        ret = self.changed_flag
        self.changed_flag = False
        return ret


class ParkingMonitor:
    def __init__(self, objects_of_interest: list):
        self.crop = (0, 0, 4056, 3040)
        self.args = self.get_args()

        self.parking_regions_extraction()
        self.objects_of_interest = objects_of_interest
        self.parking_counter = [0 for pt in self.parking_pts]

        # This must be called before instantiation of Picamera2
        self.imx500 = IMX500(self.args.model)
        self.detections = Detections(self.imx500)

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.args.fps, "ScalerCrop": self.crop}, buffer_count=28
        )
        self.picam2.start(config, show_preview=True)
        self.picam2.pre_callback = self.run

        self.parking_result = ParkingResult(len(self.parking_counter))
        signal.signal(signal.SIGINT, shutdown_handler)

    def parking_regions_extraction(self):
        """
        Extract parking regions from json file.

        Args:
                json_file (str): file that have all parking slot points
        """
        with open(self.args.json_file, "r") as json_file:
            configuration = json.load(json_file)
            if "areas" in configuration:
                parking_pts = configuration["areas"]
            if "crop_settings" in configuration:
                crop_area = configuration["crop_settings"]
                self.crop = (
                    crop_area["x"],
                    crop_area["y"],
                    crop_area["w"],
                    crop_area["h"],
                )
            if len(parking_pts) > 0:
                self.parking_pts = parking_pts
            else:
                raise Exception("Please ensure there are parkings to check")
            return parking_pts

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, required=True, help="Path of the model")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        parser.add_argument(
            "--json-file", type=str, required=False, default="config.json", help="Json file containing bboxes of parkings"
        )
        parser.add_argument(
            "--box-min-confidence", type=float, default=0.3, help="Confidence threshold for bounding box predictions"
        )
        parser.add_argument(
            "--iou-threshold",
            type=float,
            default=0.7,
            help="IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS)",
        )
        parser.add_argument(
            "--max-out-dets", type=int, default=300, help="Maximum number of output detections to keep after NMS"
        )
        self.args = parser.parse_args()
        self.BOX_MIN_CONFIDENCE = self.args.box_min_confidence
        self.IOU_THRESHOLD = self.args.iou_threshold
        self.MAX_OUT_DETS = self.args.max_out_dets

        return self.args

    def draw_parking_areas(self, request: CompletedRequest, stream="main"):
        with MappedArray(request, stream) as m:
            results = []
            for i, parking in enumerate(self.parking_pts):
                pts = np.array(parking["points"]).reshape(-1, 1, 2)
                if self.parking_counter[i] > 0:
                    text = f"{i}:O"
                    c = RED
                    occupied = True
                else:
                    text = f"{i}:F"
                    c = GREEN
                    occupied = False

                cv2.putText(
                    m.array,
                    text,
                    (parking["points"][3][0], parking["points"][3][1] + 10),
                    FONT,
                    0.5,
                    c,
                    1,
                )

                cv2.polylines(m.array, [pts], True, c, 2)
                self.parking_counter[i] = 0

    def ai_draw_parking_mgt(self, request: CompletedRequest, detections, stream="main"):
        with MappedArray(request, stream) as m:
            metadata = request.get_metadata()
            for i, parking in enumerate(self.parking_pts):  # Loop for number of parkings
                parking_reshape = np.array(parking["points"], dtype=np.int32).reshape((-1, 1, 2))
                if detections.bbox is not None:
                    if len(detections.bbox) > 0:
                        detections.tracker_id = detections.format_track()
                        for j, box in enumerate(detections.bbox):  # Loop for number of detections
                            if detections.class_id[j] in self.objects_of_interest:
                                obj_scaled = self.imx500.convert_inference_coords(box, metadata, self.picam2)
                                box = (obj_scaled[0], obj_scaled[1], obj_scaled[2], obj_scaled[3])
                                x_center = int((box[0] + (box[0] + box[2])) / 2)
                                y_center = int((box[1] + (box[1] + box[3])) / 2)

                                cv2.putText(
                                    m.array,
                                    f"{str(detections.tracker_id[j])},{int(detections.class_id[j])},{detections.confidence[j]:1.2f}",
                                    (x_center - 15, y_center + 15),
                                    FONT,
                                    0.5,
                                    GREEN,
                                    1,
                                )

                                # Calc to see if center person point is in a parking box
                                cv2.circle(m.array, (x_center, y_center), 2, GREEN, 2)
                                self.match_parking_w_object(
                                    parking, x_center, y_center, i, detections.class_id[j], detections.tracker_id[j]
                                )
                        if self.parking_counter[i] == 0:
                            self.parking_result.reset(i)


    def match_parking_w_object(self, area, x_center, y_center, i, class_id, track_id):
        area_reshape = np.array(area["points"], dtype=np.int32).reshape((-1, 1, 2))
        dist = cv2.pointPolygonTest(area_reshape, (x_center, y_center), False)
        occupied = False
        if dist >= 0:
            self.parking_counter[i] = self.parking_counter[i] + 1
            self.parking_result.set(i,  str(class_id), track_id)

    def run(self, request: CompletedRequest):
        # Update the detections and tracker from current frame.
        self.detections.update(request.get_metadata())

        # Annotate the pre defines parking areas
        self.draw_parking_areas(request)
        # Annotate the cars and calculate the parking state.
        self.ai_draw_parking_mgt(request, self.detections)
        if self.parking_result.changed():
            print(f"{self.parking_result}\n")


def start_parking_monitor_demo():
    cv2.startWindowThread()
    qm = ParkingMonitor(objects_of_interest=[2, 5, 7])
    while True:
        cv2.waitKey(30)


if __name__ == "__main__":
    start_parking_monitor_demo()

    exit()
