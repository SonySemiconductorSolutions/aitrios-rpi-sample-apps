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

import cv2
import numpy as np

from modlib.devices import AiCamera
from modlib.apps.annotate import ColorPalette, Annotator
from modlib.models.model import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_cls


from fsm import FiniteStateMachine, DEFAULT_SETTINGS


class SolderPointClassifier(Model):
    def __init__(self):
        super().__init__(
            model_file="./network/network.rpk",
            model_type=MODEL_TYPE.RPK_PACKAGED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        self.labels = np.genfromtxt("./network/labels.txt", dtype=str, delimiter="\n")

    def post_process(self, output_tensors):
        return pp_cls(output_tensors)


def start():
    device = AiCamera()
    model = SolderPointClassifier()
    device.deploy(model)
    annotator = Annotator(color=ColorPalette.default(), thickness=1, text_thickness=1, text_scale=0.4)

    fsm = FiniteStateMachine(DEFAULT_SETTINGS)
    good_ctr = 0
    bad_ctr = 0
    last_result = None
    time_stamp = None
    notification_NG_n_frames = 0
    color_map = [
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (0, 255, 255),  # Yellow
        (200, 200, 200) # Gray
    ]

    with device as stream:
        for frame in stream:
            idx = frame.detections.class_id[0]
            ai_result = {
                "C": idx,
                "label": model.labels[idx],
                "P": frame.detections.confidence[0],
                "T": frame.timestamp,
            }

            result = fsm.tick(ai_result)
       
            if result:
                if "scan_result" in result:
                    print("panel_info", result["scan_result"])
                    notification_NG_n_frames = 30
                    ng_status = result["scan_result"]["VERDICT"]
                    if ng_status == 2:
                        bad_ctr += 1
                    if ng_status == 1:
                        good_ctr += 1

                if "line_info" in result:
                    print("line_info", result["line_info"])
            else:
                if notification_NG_n_frames:
                    notification_NG_n_frames -= 1

            if notification_NG_n_frames:
                if ng_status == 2:
                    last_result = "bad"
                    time_stamp = fsm.last_frame['T']
                    cv2.circle(frame.image,(560,15), 15, (0,0,255), cv2.FILLED)
                if ng_status == 1:
                    last_result = "good"
                    time_stamp = fsm.last_frame['T']
                    cv2.circle(frame.image,(560,15), 15, (0,255,0), cv2.FILLED)

            good_text = f"Good: {good_ctr}"
            bad_text = f"Bad: {bad_ctr}"
            last_result_text = f"Last Result: {last_result}"
            time_stamp_text = f"Time stamp: {time_stamp}"

            all_text = [good_text, bad_text, last_result_text, time_stamp_text]
            for i, text in enumerate(all_text):
                annotator.set_label(
                    image=frame.image,
                    x=int(50),
                    y=int(30 + 35 * (i + 1)),
                    color= color_map[i],
                    label=text
                )

            frame.display()


if __name__ == "__main__":
    start()
