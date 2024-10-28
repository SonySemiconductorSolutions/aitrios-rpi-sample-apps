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
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet
from picamera2.devices.imx500.postprocess import scale_boxes, scale_coords, COCODrawer

from workout_monitor.matcher.object_event import ObjectEventLogic, set_settings as object_event_settings
from workout_monitor.matcher.example.utils import *
from workout_monitor.matcher.tracker_helper import *
from workout_monitor.tracker import BYTETracker

last_boxes = None
last_scores = None
last_keypoints = None

from typing import List, Tuple, Iterator, Union

class ToTrack:
    output_results: np.ndarray
    img_info: Tuple[int, int]
    img_size: Tuple[int, int]

class WorkoutExample:
    def __init__(self):
        self.args = self.get_args()
        
        pose_up_angle=145.0,
        pose_down_angle=100.0,
        
        # Image and line thickness
        self.thickness = 2
        self.max_det = 1

        # Keypoints and count information
        self.poseup_angle = pose_up_angle
        self.posedown_angle = pose_down_angle
        self.threshold = 0.001

        # Store stage, count and angle information
        self.exercise_type = self.args.exercise

        self.keypoint_check = self.get_ktps()
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
                            default=0.5, help="Confidence threshold for bounding box predictions")
        parser.add_argument("--keypoint-min-confidence", type=float,
                            default=0.5, help="Minimum confidence required to keypoint")
        parser.add_argument("--iou-threshold", type=float, default=0.7,
                            help="IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS)")
        parser.add_argument("--max-out-dets", type=int, default=300,
                            help="Maximum number of output detections to keep after NMS")
        parser.add_argument("--exercise", type=str, default="pushup", help="Exercise group can be pullup, pushup, abworkout, squat")                    
        return parser.parse_args()
        
    def get_ktps(self):
        if self.exercise_type in {"pullup", "pushup"}:
            return [6,8,10]        
        if self.exercise_type == "squat":
            return [11,13,15]
        if self.exercise_type == "abworkout":
            return [5,11,13]
        else:
            raise Exception("Please ensure exercise is a valid option")
        
    def run(self,request: CompletedRequest):
        boxes, scores, keypoints = self.ai_output_tensor_parse(request.get_metadata())
        self.ai_draw_ai_gym(request, boxes, scores, keypoints)
        
    def format_track(self,boxes,scores):
        """Format and get the tracker IDs for the detected objects"""
        out = ToTrack()
        out.img_info = [480,640]
        out.img_size = [480,640]
        out.output_results = np.hstack((np.array(boxes), scores[:,np.newaxis]))
        
        tracks = self.tracker.update(out)
        tracker_id =  match_detections_with_tracks(boxes, tracks)
        return tracker_id
        
    def ai_output_tensor_parse(self,metadata: dict):
        """Parse the output tensor into a number of detected objects, scaled to the ISP out."""
        global last_boxes, last_scores, last_keypoints
        np_outputs = self.imx500.get_outputs(metadata=metadata, add_batch=True)
        if np_outputs is not None:
            keypoints, scores, boxes = postprocess_higherhrnet(outputs=np_outputs,
                                                               img_size=(480,640),
                                                               img_w_pad=(0, 0),
                                                               img_h_pad=(0, 0),
                                                               detection_threshold=0.3,
                                                               network_postprocess=True)
            
            if scores is not None and len(scores) > 0:
                last_keypoints_t, last_boxes_t, last_scores_t = [], [], [] 
                for i,box in enumerate(boxes):
                    if scores[i] <= 0.3:
                        continue
                    else:
                        last_keypoints_t.append(keypoints[i])
                        last_boxes_t.append(boxes[i])
                        last_scores_t.append(scores[i])
                if last_scores_t is not None and len(last_scores_t) > 0:
                    last_keypoints = np.reshape(np.stack(last_keypoints_t, axis=0), (len(last_scores_t), 17, 3))
                    last_boxes = [np.array(b) for b in last_boxes_t]
                    last_boxes = scale_boxes(np.array(last_boxes), 1, 1, 480, 640, True, False)
                    last_scores = np.array(last_scores_t)
        return last_boxes, last_scores, last_keypoints
    

    def ai_draw_ai_gym(self,request: CompletedRequest, boxes, scores, keypoints, stream='main'):
        """Draw the detections for this request onto the ISP output."""

        with MappedArray(request, stream) as m:   
            Keypoints_results =[]
            b = self.imx500.get_roi_scaled(request)
            cv2.putText(m.array, "ROI", (b[0] + 5, b[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.rectangle(m.array, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), (255, 0, 0, 0))
            
            metatdata = request.get_metadata()
            if boxes is not None:
                if len(boxes) > 0 and len(scores) > 0:
                    tracker_id = self.format_track(boxes,scores)
                    for i,box in enumerate(boxes):
                        #Format model output into universal structure
                        Keypoints = [] 
                        Boxes = np.array([box[0],box[1],box[2],box[3],1,np.zeros(scores.shape)[i]])
                        for j in range(len(keypoints[i])):
                            x, y, conf = self.get_point(keypoints[i],j,metatdata)
                            P = [x,y,conf]
                            Keypoints.append(P)
                        Keypoints = np.array(Keypoints)
                        Keypoints_results.append(Keypoints)
                    im0 = self.exersice_counter(m.array, Boxes,Keypoints_results,tracker_id)

    def parse_settings(self,settings_dict):
        """Setup settings for model"""
        object_event_settings(settings_dict)
        model_fpk = os.path.abspath(settings_dict["IMX_CONFIG"]["NETWORK_FPK_PATH"])

        confidence_threshold = settings_dict["CONFIDENCE_THRESHOLD"]

        return settings_dict

    def get_point(self,keypoints, index,metadata):
        """Scale the point to image coordinates for drawing on image"""
        y0, x0 = keypoints[index][1], keypoints[index][0]
        y0 = max(0, y0)
        x0 = max(0, x0)
        return int(x0), int(y0), int(keypoints[index][2])

    def exersice_counter(self, img, Boxes,Keypoints,tracker_id):
        """
        Function used to count the gym reps of chosen exercise.
        """

        self.scene = img
        self.lw = max(round(sum(img.shape) / 2 * 0.003), 2)
        self.sf = self.lw / 3  # font scale
        
        for i, person in enumerate(Keypoints):
            if i+1 > self.max_det: #Check to see if max detetcions reached
                continue
            ID = tracker_id[i]
            #if ID == -1: #tracker has a few issues with keeping consistent ID 
                #continue
            self.keypoints = [np.array(Keypoints[i])]
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
                if self.exercise_type in {"pushup", "pullup", "abworkout", "squat"}:
                    self.angle[itr] = round(self.estimate_angle(
                        p[int(self.keypoint_check[0])],
                        p[int(self.keypoint_check[1])],
                        p[int(self.keypoint_check[2])],
                    ),3)
                    self.scene = self.draw_focus_points(p, self.keypoint_check)
                    # Check and update pose stages and counts based on angle
                    if self.exercise_type in {"abworkout", "pullup"}:
                        if self.angle[itr] > self.poseup_angle:
                            self.stage[itr] = "down"
                        if self.angle[itr] < self.posedown_angle and self.stage[itr] == "down":
                            self.stage[itr] = "up"
                            self.count[itr] += 1

                    elif self.exercise_type in {"pushup", "squat"}:
                        if self.angle[itr] > self.poseup_angle:
                            self.stage[itr] = "up"
                        if self.angle[itr] < self.posedown_angle and self.stage[itr] == "up":
                            self.stage[itr] = "down"
                            self.count[itr] += 1

                    angle_text, count_text, stage_text = (f" {self.angle[itr]:.2f}", f"Steps : {self.count[itr]}", f" {self.stage[itr]}")
                    center_kpt=p[1]
                    txt_color = (255,255,255)
                    
                    # Draw angle
                    (angle_text_width, angle_text_height), _ = cv2.getTextSize(angle_text, 0, self.sf, self.thickness)
                    angle_text_position = (int(center_kpt[0]), int(center_kpt[1]))
                    cv2.putText(self.scene, angle_text, angle_text_position, 0, self.sf, txt_color, self.thickness)

                    # Draw Counts
                    (count_text_width, count_text_height), _ = cv2.getTextSize(count_text, 0, self.sf, self.thickness)
                    count_text_position = (angle_text_position[0], angle_text_position[1] + angle_text_height + 20)
                    cv2.putText(self.scene, count_text, count_text_position, 0, self.sf, txt_color, self.thickness)

                    # Draw Stage
                    (stage_text_width, stage_text_height), _ = cv2.getTextSize(stage_text, 0, self.sf, self.thickness)
                    stage_text_position = (int(center_kpt[0]), int(center_kpt[1]) + angle_text_height + count_text_height + 40)
                    cv2.putText(self.scene, stage_text, stage_text_position, 0, self.sf, txt_color, self.thickness)
                    
                # Draw keypoints
                self.visualize_keypoints(p, keypoint_line=True)
                
    def visualize_keypoints(self, keypoints, keypoint_line=True, conf_thres=0.15):
        """
        Plot keypoints on the image.
        """
        self.skeleton = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13], [12, 14], [13, 15], [14, 16]]
        self.skel_colours = np.array([[35, 104, 222],[35, 222, 203],[51, 222, 35],[144, 222, 35],],dtype=np.uint8,)
        self.limb_color = self.skel_colours[[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]
        self.point_color = self.skel_colours[[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3]]
        keypoint_n, dim_n = keypoints.shape
    
        for i, k in enumerate(keypoints):
            color_k = [int(x) for x in self.point_color[i]]
            x_coord, y_coord = k[0], k[1]
            if x_coord % 640 != 0 and y_coord % 640 != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < conf_thres:
                        continue
                cv2.circle(self.scene, (int(x_coord), int(y_coord)),1, color_k, -1)
                
        for i, skel_joint in enumerate(self.skeleton):
            P1 = (int(keypoints[(skel_joint[0]), 0]), int(keypoints[(skel_joint[0]), 1]))
            P2 = (int(keypoints[(skel_joint[1]) , 0]), int(keypoints[(skel_joint[1]), 1]))
            
            if P1[0] % 640 == 0 or P1[1] % 640 == 0 or P1[0] < 0 or P1[1] < 0:
                continue
            if P2[0] % 640 == 0 or P2[1] % 640 == 0 or P2[0] < 0 or P2[1] < 0:
                continue
            cv2.line(self.scene, P1, P2, [int(x) for x in self.limb_color[i]], thickness=2)
        
    @staticmethod    
    def estimate_angle(p1, p2, p3):
        """
        Calculate the angle of the chosen keypoint exercise
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
        a1 = p1 - p2
        a2 = p3-p2
        cos_a = np.dot(a1,a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))
        keypoint_angle = np.degrees(np.arccos(cos_a))
        if keypoint_angle > 180.0:
            keypoint_angle = 360 - keypoint_angle
        return keypoint_angle
        
    def draw_focus_points(self, keypoints, point_check):
        """
        Draw points of focus for the exercise to visulaize them better
        """
        for i, point in enumerate(keypoints):
            if i in point_check:
                if point[0] ==0 and point[1] ==0:
                    continue 
                x1, y1 = point[0], point[1]
                cv2.circle(self.scene, (int(x1), int(y1)), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        return self.scene
        
def start_workout_demo():
    cv2.startWindowThread()
    wo = WorkoutExample()
    wo.run
    while True:
        cv2.waitKey(30)

if __name__ == "__main__":
    start_workout_demo()
    exit()

