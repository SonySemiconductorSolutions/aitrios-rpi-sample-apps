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

from modlib.devices import AiCamera
from modlib.models.zoo import Posenet
from modlib.apps.tracker.byte_tracker import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator, Color
from modlib.apps.calculate import estimate_angle
from modlib.models.results import Poses
from modlib.devices.frame import Frame

class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False
    
class BoxLift:
    """
    A class responsible for parsing new detections, calculating the angles of a correct 'lift',
    and ensuring that each person's state is updated accordingly. Storing all the data need to 
    track and calculate a lift over time. 
    ```
    """
    def __init__(self,points_to_check: list):
        """
        Args:
            points_to_check: List of Lists that contains the points of the key angles in calculating a correct lift
        """
        self.tracked_IDs = {} 
        self.focus_points = points_to_check
        self.pickup_correct = 0
        self.pickup_incorrect = 0
        
    def update(self, detections: Poses ,frame: Frame):
        '''
        Args:
            detections: The set of Poses detected in the frame
            frame: The current frame from camera
        '''
        if len(self.tracked_IDs) > 100:  # Pop unused items
            self.tracked_IDs.pop(list(self.tracked_IDs.keys())[0])
            
        for k, s, _, b, t in detections:
            if t not in self.tracked_IDs:
                self.tracked_IDs[t] = [0, 0, 0, 0, "-",(200,200,200),0] #(back, leg, stand, count, stage, color, temp)
            tracked_data = self.tracked_IDs[t]
            for i, check_angle in enumerate(self.focus_points):
                angle = estimate_angle(k,check_angle,frame.width ,frame.height)
                if angle is None:
                    angle = tracked_data[i]
                else:
                    tracked_data[i] = angle
            self.tracked_IDs[t] = self.lift_check(tracked_data)
            
    def lift_check(self, tracked_data: list):
        '''
        Logic to check if all the calculated angles are what a correct lift should be.
        Args:
            tracked_data: data of tracked person
        '''
        if tracked_data[2] > 160: #If person is standing
            if tracked_data[4] == "picking up" or tracked_data[4] == "incorrect": #If person was lifting incorrect
                self.pickup_incorrect += 1
                tracked_data[5] = (0,0,255)
            elif tracked_data[4] == "correct": #If person was lifing correct
                self.pickup_correct += 1
                tracked_data[3] += 1
                tracked_data[5] = (0,255,0)
            tracked_data[4] = "standing"
            
        else: 
            if tracked_data[0] >160 and tracked_data[1] >130 : 
                tracked_data[4] = "standing"
            else: 
                if tracked_data[4] == "picking up":
                    if tracked_data[0] <160 and tracked_data[1] <125: #Check of lift correct
                        tracked_data[4] = "correct"
                    else:
                        tracked_data[4] = "incorrect"
                elif tracked_data[4] == "correct":
                    tracked_data[4] = "correct"
                elif tracked_data[4] == "incorrect": #Chance for person to recorrect lift
                    if tracked_data[0] <160 and tracked_data[1] <125:
                        tracked_data[4] = "correct"
                    else:
                        tracked_data[4] = "incorrect"
                else:
                    if (tracked_data[1] - tracked_data[6]) > 12: #difference between last leg angle to see if lif has started
                        tracked_data[4] = "picking up"

                    else:
                        tracked_data[4] = "bending"
        tracked_data[6] = tracked_data[1] # store leg angle in temp value
        return tracked_data


def draw_focus_points(frame, keypoints, point_check):
    """
    Draw points of focus for the exercise to visulaize them better
    """
    image = frame.image
    for i in range(17):
        if i in (point for angle in point_check for point in angle):
            if keypoints[i][0] == 0.0 and keypoints[i][1] == 0.0:
                continue
            x = int(keypoints[i][0] * frame.width)
            y = int(keypoints[i][1] * frame.height)
            cv2.circle(image, (x, y), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    return image

def start_lift_demo():
    #-----Camera and AI setup-----
    device = AiCamera()
    model = Posenet()
    device.deploy(model)
    
    tracker = BYTETracker(BYTETrackerArgs())
    annotator = Annotator(
        color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
    )
    lift = BoxLift(points_to_check=[[5,11,13],[11,13,15],[5,11,15]]) #back, legs, standing points
    with device as stream:
        for frame in stream:
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > 0.2]
            #-----Tracker-----
            detections = tracker.update(frame, detections)
            #-----Application Logic-----
            lift.update(detections, frame)
            #-----Display Annotations-----
            for k, s, _, b, t in detections:
                ID_data = lift.tracked_IDs[t]
                
                # Draw Reps
                annotator.set_label(
                    image=frame.image,
                    x=int((b[0]- (b[0] - b[2]))*frame.width),
                    y=int(b[1]*frame.height + 30),
                    color= lift.tracked_IDs[t][5],
                    label="correct: " + str(ID_data[3]),
                )
                # Draw Stage
                annotator.set_label(
                    image=frame.image,
                    x=int((b[0]- (b[0] - b[2]))*frame.width),
                    y=int(b[1]*frame.height + 60),
                    color= lift.tracked_IDs[t][5],
                    label="Stage: " + str(ID_data[4]),
                )

                # Draw Focus Points
                frame.image = draw_focus_points(frame, k, lift.focus_points)

            frame.image = annotator.annotate_keypoints(
                frame=frame,
                poses=detections,
                keypoint_color = Color(0, 255, 0), 
                line_color = Color(0, 255, 0), 
                keypoint_score_threshold=0.2,
            )
            frame.display()

if __name__ == "__main__":
    start_lift_demo()
    exit()
