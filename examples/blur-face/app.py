"""
  Copyright 2026 Sony Semiconductor Solutions Corp. All rights reserved.

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

from modlib.apps import blur
from modlib.devices import AiCamera
from modlib.models.zoo import Posenet

def start_blur_face_demo():
    device = AiCamera()
    model = Posenet()
    device.deploy(model)

    threshold = 0.1

    with device as stream:  
        for frame in stream:
            #-----Detection Filtering-----
            detections = frame.detections[frame.detections.confidence > threshold]
            # get the face keypoint_scores and check if their average is greater than threshold
            filter = [sum(d[:5]) / 5 > threshold for d in detections.keypoint_scores]
            filtered_detections = detections[filter]
            
            #-----Display Annotations-----
            blur.blur_face(frame, filtered_detections, intensity=50, padding=10) 
            frame.display()

if __name__ == "__main__":
    start_blur_face_demo()
