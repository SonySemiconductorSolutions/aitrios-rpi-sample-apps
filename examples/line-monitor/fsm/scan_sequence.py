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

import logging
from datetime import datetime

# 	STATUS_INIT,  0
# 	STATUS_OK,    1
# 	STATUS_BAD,   2
# 	STATUS_ERROR, 3

scan_result = {"STATUS_INIT": 0, "STATUS_OK": 1, "STATUS_BAD": 2, "STATUS_ERROR": 3}
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARNING)


class ScanSequence:
    def __init__(self, frames, last_result):
        self.frames = frames
        self.trimmed_frames = None
        self.start_time = None
        self.end_time = None
        self.elapsed_seconds = 0.0
        if last_result and "scan_result in last_result":
            self.last_result = last_result["scan_result"]
        else:
            self.last_result = last_result

    def trim(self):
        while self.frames and self.frames[0]["mapped_class"] == "BACKGROUND":
            self.frames.popleft()

        while self.frames and self.frames[-1]["mapped_class"] == "BACKGROUND":
            self.frames.pop()

        return self.frames

    def analyze(self, settings):
        takt_time = 0
        panel_count = 0
        self.trimmed_frames = self.trim()
        if len(self.trimmed_frames) == 0:
            logging.warning("No scanned frames stored. Returning..")
            return None

        self.elapsed_seconds = ScanSequence.elapsed_time(self.trimmed_frames)
        score = ScanSequence.calculate_result(self.trimmed_frames, settings)
        logging.info(f"Scan duration: {self.elapsed_seconds}")
        result = {}
        result["SCAN_TIME_S"] = self.elapsed_seconds
        result["T"] = self.trimmed_frames[0]["T"]

        if self.last_result:
            takt_time = ScanSequence.time_diff(self.last_result["T"], result["T"])
            panel_count = self.last_result["PANEL_COUNT"] + 1
        result["TAKT_TIME"] = takt_time
        result["PANEL_COUNT"] = panel_count
        result["SUBCLASS"] = score["SUBCLASS"]

        if score["RESULT_OK"] == 0 and score["RESULT_BAD"] == 0:
            result["VERDICT"] = scan_result["STATUS_ERROR"]
        elif score["RESULT_OK"] > score["RESULT_BAD"]:
            result["VERDICT"] = scan_result["STATUS_OK"]
        else:
            result["VERDICT"] = scan_result["STATUS_BAD"]
            if "INPUT" in score:
                result["INPUT"] = score["INPUT"]
        ret = {}
        ret["scan_result"] = result
        ret["scan_details"] = score
        return ret

    @staticmethod
    def time_diff(ts1, ts2):
        time_a = datetime.fromisoformat(ts1)
        time_b = datetime.fromisoformat(ts2)
        elapsed_seconds = time_b - time_a
        return elapsed_seconds.total_seconds()

    def elapsed_time(frames):
        start = frames[0]["T"]
        end = frames[-1]["T"]

        elapsed_seconds = ScanSequence.time_diff(start, end)
        logging.debug(f"Elapsed scan time: {elapsed_seconds} seconds.")
        return elapsed_seconds

    @staticmethod
    def class_is_background(mapped_class):
        if mapped_class == "BACKGROUND":
            return True
        elif (
            mapped_class == "OBJECT"
            or mapped_class == "RESULT_OK"
            or mapped_class == "RESULT_BAD"
        ):
            return False
        else:
            return None

    @staticmethod
    def calculate_result(frames, settings):
        result = {}
        result["RESULT_OK"] = 0
        result["RESULT_BAD"] = 0
        result["SUBCLASS"] = -1
        if "INPUT" in frames[0]:
            result["INPUT"] = frames[0]["INPUT"]
            result["INPUT_C"] = frames[0]["C"]
            result["INPUT_P"] = frames[0]["P"]
        last_score = ""
        accumulated_probability = 0
        for frame in frames:
            if not (
                frame["mapped_class"] == "RESULT_OK"
                or frame["mapped_class"] == "RESULT_BAD"
            ):
                continue
            P = int(frame["P"] * 100)
            if frame["mapped_class"] == last_score:
                if P > settings["PROBABILITY_THRESHOLD_PERCENT"]:
                    accumulated_probability = accumulated_probability + (1 * P)

                    if accumulated_probability > result[last_score]:
                        result[last_score] = accumulated_probability
                        result["SUBCLASS"] = frame["C"]
                        if "INPUT" in frame:
                            result["INPUT"] = frame["INPUT"]
                            result["INPUT_C"] = frame["C"]
                            result["INPUT_P"] = frame["P"]
            else:
                last_score = frame["mapped_class"]
                accumulated_probability = 0
        logging.info("Final verdict:")
        for k, v in result.items():
            logging.info(f"\t{k} is: {v}")
        return result
