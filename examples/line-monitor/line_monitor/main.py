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

import time
import signal
import sys
import logging

from line_monitor.aicamera import IMX500Data, AbstractWorker, ClassificationDecoder
from line_monitor.util import JSONFileManager, Annotator
from line_monitor.fsm import FiniteStateMachine, DEFAULT_SETTINGS

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)


class LineMonitor(AbstractWorker):
    def __init__(self):
        self.settings = JSONFileManager(DEFAULT_SETTINGS).get_json_data()
        self.fsm_instance = FiniteStateMachine(self.settings)
        self.decoder = ClassificationDecoder(labels_filename=self.settings["AI"]["Labels"])
        self.annotator = Annotator()
        self.ai_camera = IMX500Data(self.settings, self)

        self.ai_camera.start_stream()
        signal.signal(signal.SIGINT, self.signal_handler)

    def do_work_callback(self, *args, **kwargs):
        tensors, cv_map = args[0], args[1]

        ai_result = self.decoder.get_prediction_details(tensors["Imx500OutputTensor"], tensors["CaptureTime"])
        result = self.fsm_instance.tick(ai_result)

        if result:
            if "scan_result" in result:
                print("panel_info", result["scan_result"])
            if "line_info" in result:
                print("line_info", result["line_info"])

        annotation_info = self.fsm_instance.info() + [f"FPS: {self.ai_camera.fps():2.1f}"]
        self.annotator.annotate(cv_map, annotation_info)

    def loop(self):
        while True:
            time.sleep(1)

    def signal_handler(self, sig, frame):
        logging.warning("Captured Ctrl+C! Shutting down gracefully.")
        self.ai_camera.stop_stream()
        sys.exit(0)


def start():
    lm = LineMonitor()
    lm.loop()


if __name__ == "__main__":
    start()
