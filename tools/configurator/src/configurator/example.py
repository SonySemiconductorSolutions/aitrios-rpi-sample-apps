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
from modlib.models.zoo import EfficientNetB0
from modlib.apps.annotate import Annotator, Color
from . import CVMenuStateMachine, config


device = AiCamera()
model = EfficientNetB0()
device.deploy(model)
gs = CVMenuStateMachine()
roi = config.get_roi()
print(f"Configuration is: {config} and roi: {roi}. type: {type(roi)}")

device.set_input_tensor_cropping(tuple(roi))
device.set_image_cropping(tuple(roi))


def main():
    annotator = Annotator(Color.red, thickness=2, text_scale=0.5)
    with device as stream:
        for frame in stream:

            for i, label in enumerate([model.labels[id] for id in frame.detections.class_id[:3]]):
                text = f"{i+1}. {label}: {frame.detections.confidence[i]:.2f}"
                gs.tick(frame)
                cv2.putText(frame.image, text, (50, 30 + 40 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 0, 100), 2)
                for area in config.get_areas():
                    frame.image = annotator.annotate_area(frame, area, (0, 200, 200))

            frame.display()


if __name__ == "__main__":
    main()
