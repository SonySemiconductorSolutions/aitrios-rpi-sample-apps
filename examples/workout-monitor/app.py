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

import argparse
import cv2
import numpy as np

from modlib.devices import AiCamera
from modlib.models.zoo import Posenet
from modlib.apps.tracker.byte_tracker import BYTETracker
from modlib.apps.annotate import ColorPalette, Annotator
from typing import List, Optional
from modlib.models import Poses
from modlib.apps.calculate import estimate_angle

from modlib.devices.frame import Frame, IMAGE_TYPE

class BYTETrackerArgs:
	track_thresh: float = 0.25
	track_buffer: int = 30
	match_thresh: float = 0.8
	aspect_ratio_thresh: float = 3.0
	min_box_area: float = 1.0
	mot20: bool = False

# ---------------------------------------------------------------------

def custom_annotate_poses(
        frame: Frame,
        poses: Poses,
        keypoint_radius: Optional[int] = 4,
        keypoint_color: Optional[List] = [0, 255, 0],
        line_color: Optional[List] = [0, 255, 0],
        keypoint_score_threshold: Optional[float] = 0.5,
    ) -> np.ndarray:
        
        if not isinstance(poses, Poses):
            raise ValueError("Detections must be of type Poses.")

        # NOTE: Compensating for any introduced modified region of interest (ROI)
        # to ensure that detections are displayed correctly on top of the current `frame.image`.
        if frame.image_type != IMAGE_TYPE.INPUT_TENSOR:
            poses.compensate_for_roi(frame.roi)
        
        special_line_color = [255, 255, 0]

        # first element in the tuple is color code
        # 0 -> line_color, 1 -> special_line_color
        skeleton = [
            (0, 5, 6),
            (1, 11, 12),
            (0, 5, 7),
            (0, 7, 9),
            (1, 5, 11),
            (1, 11, 13),
            (0, 13, 15),
            (0, 6, 8),
            (0, 8, 10),
            (1, 6, 12),
            (1, 12, 14),
            (0, 14, 16),
        ]

        h, w, _ = frame.image.shape

        def draw_keypoints(poses, image, pose_idx, keypoint_idx, w, h, threshold=keypoint_score_threshold):
            if poses.keypoint_scores[pose_idx][keypoint_idx] >= threshold:
                y = int(poses.keypoints[pose_idx][keypoint_idx][1] * h)
                x = int(poses.keypoints[pose_idx][keypoint_idx][0] * w)
                cv2.circle(image, (x, y), keypoint_radius, keypoint_color, -1)

        def draw_line(poses, image, pose_idx, keypoint1, keypoint2, w, h, color_index, threshold=keypoint_score_threshold ):
            color = line_color
            if (
                poses.keypoint_scores[pose_idx][keypoint1] >= threshold
                and poses.keypoint_scores[pose_idx][keypoint2] >= threshold
            ):
                y1 = int(poses.keypoints[pose_idx][keypoint1][1] * h)
                x1 = int(poses.keypoints[pose_idx][keypoint1][0] * w)
                y2 = int(poses.keypoints[pose_idx][keypoint2][1] * h)
                x2 = int(poses.keypoints[pose_idx][keypoint2][0] * w)
                if color_index == 1:
                    color = special_line_color

                cv2.line(image, (x1, y1), (x2, y2), color, 2)

        for i in range(poses.n_detections):
            if poses.confidence[i] > keypoint_score_threshold:
                # Draw keypoints
                for j in range(17):
                    draw_keypoints(poses, frame.image, i, j, w, h)

                # Draw skeleton lines
                for color_index, keypoint1, keypoint2 in skeleton:
                    draw_line(poses, frame.image, i, keypoint1, keypoint2, w, h, color_index)

        return frame.image


def draw_focus_points(frame, keypoints, point_check):
	"""
	Draw points of focus for the exercise to visulaize them better
	"""
	image = frame.image
	for i in range(17):
		if i in point_check:
			if keypoints[i][0] == 0.0 and keypoints[i][1] == 0.0:
				continue
			x = int(keypoints[i][0] * frame.width)
			y = int(keypoints[i][1] * frame.height)
			cv2.circle(image, (x, y), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
	return image


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--exercise",
		type=str,
		default="pushup",
		help="Exercise group can be pullup, pushup, abworkout, squat",
	)
	return parser.parse_args()


def get_exercise_ktps(exercise_name):
	# To add new exercise types add it here
	if exercise_name in {"pullup", "pushup"}:
		return [6, 8, 10]
	if exercise_name == "squat":
		return [11, 13, 15]
	if exercise_name == "abworkout":
		return [5, 11, 13]
	else:
		raise Exception("Please ensure exercise is a valid option")


def start_workout_demo():
	#-----Camera and AI setup-----
	args = get_args()
	device = AiCamera()
	model = Posenet()
	device.deploy(model)

	pose_up_angle = 145.0
	pose_down_angle = 100.0
	focus_keypoints = get_exercise_ktps(args.exercise)
	tracked_IDs = {}

	tracker = BYTETracker(BYTETrackerArgs())
	annotator = Annotator(
		color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4
	)

	with device as stream:
		for frame in stream:
			#-----Detection Filtering-----
			detections = frame.detections[frame.detections.confidence > 0.3]
			#-----Tracker-----
			detections = tracker.update(frame, detections)
			#-----Application Logic-----
			for k, s, _, b, t in detections:
				if t == -1:
					continue
				if t not in tracked_IDs:
					tracked_IDs[t] = [0.0, 0, "-"]  # angle, reps, stage
				else:
					person = tracked_IDs.get(t)
					angle = estimate_angle(
						k, focus_keypoints, frame.height, frame.width
					)
					if angle is None:  # When point is (0,0) use previous good angle
						angle = person[0]

					if args.exercise in {"abworkout", "pullup"}:
						if angle > pose_up_angle:
							tracked_IDs[t] = [angle, person[1], "down"]
						if angle < pose_down_angle and person[2] == "down":
							tracked_IDs[t] = [angle, person[1] + 1, "up"]

					elif args.exercise in {"pushup", "squat"}:
						if angle > pose_up_angle:
							tracked_IDs[t] = [angle, person[1], "up"]
						if angle < pose_down_angle and person[2] == "up":
							tracked_IDs[t] = [angle, person[1] + 1, "down"]

				if len(tracked_IDs) > 100:  # Pop unused items
					tracked_IDs.pop(list(tracked_IDs.keys())[0])
				#-----Display Annotations-----
				# Draw Angle
				annotator.set_label(
					image=frame.image,
					x=int((b[0]- (b[0] - b[2]))*frame.width),
                    y=int(b[1]*frame.height),
					color= (200, 200, 200),
					label="Angle: " + str(tracked_IDs[t][0]),
				)
				# Draw Reps
				annotator.set_label(
					image=frame.image,
					x=int((b[0]- (b[0] - b[2]))*frame.width),
                    y=int(b[1]*frame.height + 30),
					color= (200, 200, 200),
					label="Reps: " + str(tracked_IDs[t][1]),
				)
				# Draw Stage
				annotator.set_label(
					image=frame.image,
					x=int((b[0]- (b[0] - b[2]))*frame.width),
                    y=int(b[1]*frame.height + 60),
					color= (200, 200, 200),
					label="Stage: " + str(tracked_IDs[t][2]),
				)

				# Draw Focus Points
				frame.image = draw_focus_points(frame, k, focus_keypoints)
			
			frame.image = custom_annotate_poses(
				frame=frame,
				poses=detections,
				keypoint_color = (0, 255, 0), 
				line_color = (0, 255, 0), 
				keypoint_score_threshold=0.3,
			)
			frame.display()


if __name__ == "__main__":
	start_workout_demo()
	exit()
