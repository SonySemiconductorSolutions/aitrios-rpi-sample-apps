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

from picamera2.devices.imx500 import IMX500
from picamera2 import Picamera2, MappedArray, CompletedRequest
from picamera2.devices.imx500.postprocess_yolov8 import postprocess_yolov8_keypoints
from picamera2.devices.imx500.postprocess import scale_boxes, scale_coords, COCODrawer

from workout_monitor.matcher.object_event import ObjectEventLogic, set_settings as object_event_settings
from workout_monitor.matcher.example.utils import *
from workout_monitor.matcher.tracker_helper import *
from workout_monitor.tracker.detectionsKeyPoint import *
from workout_monitor.tracker import BYTETracker

last_boxes = None
last_scores = None
last_keypoints = None

class WorkoutExample:
    def __init__(self):
        self.args = self.get_args()
        
        pose_up_angle=145.0,
        pose_down_angle=100.0,
        
        # Image and line thickness
        self.im0 = None
        self.tf = 2
        self.max_det = 3

        # Keypoints and count information
        self.keypoints = None
        self.poseup_angle = pose_up_angle
        self.posedown_angle = pose_down_angle
        self.threshold = 0.001

        # Store stage, count and angle information
        self.angle = None
        self.count = None
        self.stage = None
        self.pose_type = self.args.exercise

        self.kpts_to_check = self.get_ktps()
        # Visual Information
        self.view_img = False

        self.count = []
        self.angle = []
        self.stage = []
        self.IDs = []
        
        
        self.BOX_MIN_CONFIDENCE = self.args.box_min_confidence
        self.KEYPOINT_MIN_CONFIDENCE = self.args.keypoint_min_confidence
        self.IOU_THRESHOLD = self.args.iou_threshold
        self.MAX_OUT_DETS = self.args.max_out_dets
        
        self.settings_dict = self.parse_settings(settings("settings.json"))
        signal.signal(signal.SIGINT, shutdown_handler)
        
        # This must be called before instantiation of Picamera2
        self.imx500 = IMX500(self.args.model)
        self.tracker = BYTETracker(BYTETrackerArgs(self.settings_dict))

        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(controls={'FrameRate': self.args.fps}, buffer_count=28)
        self.picam2.start(config, show_preview=True)
        self.imx500.set_auto_aspect_ratio()
        self.picam2.pre_callback = self.run

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", type=str, default="networks/imx500_network_yolov8n_pose.rpk", required=True, help="Path of the model")
        parser.add_argument("--fps", type=int, default=30, help="Frames per second")
        parser.add_argument("--box-min-confidence", type=float,
                            default=0.3, help="Confidence threshold for bounding box predictions")
        parser.add_argument("--keypoint-min-confidence", type=float,
                            default=0.3, help="Minimum confidence required to keypoint")
        parser.add_argument("--iou-threshold", type=float, default=0.7,
                            help="IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS)")
        parser.add_argument("--max-out-dets", type=int, default=300,
                            help="Maximum number of output detections to keep after NMS")
        parser.add_argument("--exercise", type=str, default="pushup", help="Exercise group can be pullup, pushup, abworkout, squat")                    
        return parser.parse_args()
        
    def get_ktps(self):
        if self.pose_type in {"pullup", "pushup"}:
            return [6,8,10]        
        if self.pose_type == "squat":
            return [11,13,15]
        if self.pose_type == "abworkout":
            return [5,11,13]
        else:
            raise Exception("Please ensure exercise is a valid option")
        
    def run(self,request: CompletedRequest):
        boxes, scores, keypoints = self.ai_output_tensor_parse(request.get_metadata())
        self.ai_draw_ai_gym(request, boxes, scores, keypoints)
        
    def format_track(self,boxes,scores):
        """Format and get the tracker IDs for the detected objects"""
        out = ToTrack()
        out.img_info = [640,640]
        out.img_size = [640,640]
        out.output_results = np.hstack((np.array(boxes), scores[:,np.newaxis]))
        
        tracks = self.tracker.update(out)
        tracker_id =  match_detections_with_tracks(boxes, tracks)
        return tracker_id
        
    def ai_output_tensor_parse(self,metadata: dict):
        """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
        global last_boxes, last_scores, last_keypoints
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is not None:
            boxes, last_scores, keypoints = postprocess_yolov8_keypoints(outputs=np_outputs, conf=self.BOX_MIN_CONFIDENCE,
                                                                         iou_thres=self.IOU_THRESHOLD, max_out_dets=self.MAX_OUT_DETS)
            keypoints = np.reshape(keypoints, [keypoints.shape[0], 17, 3])
            last_keypoints = scale_coords(keypoints, 1, 1, 640, 640, True)
            last_boxes = scale_boxes(boxes, 1, 1, 640, 640, True, False)
        return last_boxes, last_scores, last_keypoints
    

    def ai_draw_ai_gym(self,request: CompletedRequest, boxes, scores, keypoints, stream='main'):
        """Draw the detections for this request onto the ISP output."""

        with MappedArray(request, stream) as m:   
            results =[]
            b = self.imx500.get_roi_scaled(request)
            cv2.putText(m.array, "ROI", (b.x + 5, b.y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(m.array, (b.x, b.y), (b.x + b.width, b.y + b.height), (255, 0, 0, 0))
            
            metatdata = request.get_metadata()
            if boxes is not None:
                if len(boxes) > 0:
                    tracker_id = self.format_track(boxes,scores)
                    
                    for i,box in enumerate(boxes):
                        #Format model output into universal structure
                        Keypoints = [] 
                        Boxes = np.array([box[0],box[1],box[2],box[3],1,np.zeros(scores.shape)[i],tracker_id[i]])
                        for j in range(len(keypoints[i])):
                            x, y = self.get_point(keypoints[i],j,metatdata)
                            P = [x,y]
                            Keypoints.append(P)
                        Keypoints = np.array(Keypoints)
                        results.append(Results(m.array,boxes=Boxes, keypoints=Keypoints))
                    im0 = self.exersice_counter(m.array, results)

    def parse_settings(self,settings_dict):
        """Setup settings for model"""
        object_event_settings(settings_dict)
        model_fpk = os.path.abspath(settings_dict["IMX_CONFIG"]["NETWORK_FPK_PATH"])

        confidence_threshold = settings_dict["CONFIDENCE_THRESHOLD"]

        return settings_dict

    def get_point(self,keypoints, index,metadata):
        """Scale the point to image coordinates for drawing on image"""
        y0, x0 = keypoints[index][1], keypoints[index][0]
        coords = y0, x0, y0 + 1, x0 + 1
        obj_scaled = self.imx500.convert_inference_coords(coords, metadata, self.picam2, stream="main")
        return obj_scaled.x, obj_scaled.y

    def exersice_counter(self, im, results):
        """
        Function used to count the gym steps.

        Args:
            im0 (ndarray): Current frame from the video stream.
            results (list): Pose estimation data.
        """

        self.im = im
        self.lw = max(round(sum(im.shape) / 2 * 0.003), 2)
        self.sf = self.lw / 3  # font scale
        
        for i in range(len(results)):
            if i+1 > self.max_det:
                continue
            ID = results[i].boxes.data[0][-1]
            self.keypoints = results[i].keypoints.data
            if ID == -1:
                continue
            if ID not in self.IDs:
                self.IDs.append(ID)
                self.count.append(0)  
                self.angle.append(0)
                self.stage.append("-")
                itr = len(self.IDs) -1
            else:
                for j,idcheck in enumerate(self.IDs):
                    if idcheck == ID:
                        itr = j 
            for ind, p in enumerate(reversed(self.keypoints)):
                # Estimate angle and draw specific points based on pose type
                if self.pose_type in {"pushup", "pullup", "abworkout", "squat"}:
                    self.angle[itr] = round(self.estimate_angle(
                        p[int(self.kpts_to_check[0])],
                        p[int(self.kpts_to_check[1])],
                        p[int(self.kpts_to_check[2])],
                    ),3)
                    self.im = self.draw_focus_points(p, self.kpts_to_check, shape=(640, 640), radius=7)
                    # Check and update pose stages and counts based on angle
                    if self.pose_type in {"abworkout", "pullup"}:
                        if self.angle[itr] > self.poseup_angle:
                            self.stage[itr] = "down"
                        if self.angle[itr] < self.posedown_angle and self.stage[itr] == "down":
                            self.stage[itr] = "up"
                            self.count[itr] += 1

                    elif self.pose_type in {"pushup", "squat"}:
                        if self.angle[itr] > self.poseup_angle:
                            self.stage[itr] = "up"
                        if self.angle[itr] < self.posedown_angle and self.stage[itr] == "up":
                            self.stage[itr] = "down"
                            self.count[itr] += 1

                    angle_text, count_text, stage_text = (f" {self.angle[itr]:.2f}", f"Steps : {self.count[itr]}", f" {self.stage[itr]}")
                    center_kpt=p[int(self.kpts_to_check[1])]
                    txt_color = (255,255,255)
                    
                    # Draw angle
                    (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, self.sf, self.tf)
                    angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
                    cv2.putText(self.im, angle_text, angle_text_position, 0, self.sf, txt_color, self.tf)

                    # Draw Counts
                    (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, self.sf, self.tf)
                    count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 20)
                    cv2.putText(self.im, count_text, count_text_position, 0, self.sf, txt_color, self.tf)

                    # Draw Stage
                    (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, self.sf, self.tf)
                    stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
                    cv2.putText(self.im, stage_text, stage_text_position, 0, self.sf, txt_color, self.tf)
                    
                # Draw keypoints
                self.kpts(p, shape=(640, 640), radius=1, kpt_line=True)
                
    def kpts(self, kpts, shape=(640, 640), radius=5, kpt_line=True, conf_thres=0.25):
        """
        Plot keypoints on the image.

        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                       for human pose. Default is True.

        Note:
            `kpt_line=True` currently only supports human pose plotting.
        """
        self.skeleton = [[16, 14],[14, 12],[17, 15],[15, 13],[12, 13],[6, 12],[7, 13],[6, 7],[6, 8],[7, 9],[8, 10],[9, 11],[2, 3],[1, 2],[1, 3]]
        self.skel_colours = np.array(
            [
                [35, 104, 222],
                [35, 222, 203],
                [51, 222, 35],
                [144, 222, 35],
            ],
            dtype=np.uint8,
        )
        self.limb_color = self.skel_colours[[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]
        self.point_color = self.skel_colours[[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim in {2, 3}
        kpt_line &= is_pose 
        for i, k in enumerate(kpts):
            color_k = [int(x) for x in self.point_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, color_k, -1)

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < conf_thres or conf2 < conf_thres:
                        continue
                if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                    continue
                if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                    continue
                cv2.line(self.im, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2)

        return self.im0
        
    @staticmethod    
    def estimate_angle(a, b, c):
        """
        Calculates the angle between 3 points in 2D space
        """

        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle
        
    def draw_focus_points(self, keypoints, indices=None, shape=(640, 640), radius=2, conf_thres=0.25):
        """
        Draw specific keypoints that are used in angle calculation.

        """

        for i, k in enumerate(keypoints):
            if i in indices:
                x_coord, y_coord = k[0], k[1]
                if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                    if len(k) == 3:
                        conf = k[2]
                        if conf < conf_thres:
                            continue
                    cv2.circle(self.im, (int(x_coord), int(y_coord)), radius, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.im
        
def start_workout_demo():
    cv2.startWindowThread()
    wo = WorkoutExample()
    wo.run
    while True:
        cv2.waitKey(30)

if __name__ == "__main__":
    start_workout_demo()
    exit()

