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

import os

generic_labels = ["BACKGROUND", "OBJECT", "RESULT_OK", "RESULT_BAD", "LEN"]
results = {"RESULT_OK": 1, "RESULT_BAD": 2}


class Labels:
    def __init__(self, mapping, path=None):
        self.path = path
        self.labels = []
        self.label_to_gen = mapping

        self.read()

    def read(self):

        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.labels = f.read().splitlines()
                index = 0
                for label in self.labels:
                    print(f"{index} : {label}")
                    index += 1
        else:
            print(f"{self.path} not found.")
            exit()

    def class_to_label(self, class_index):
        if class_index >= len(self.labels):
            return f"CLASS {class_index} OUT OF RANGE"
        else:
            return self.labels[class_index]

    def map_label(self, label):
        label = label.upper()
        if label in self.label_to_gen:
            return self.label_to_gen[label]
        else:
            # print(f"Label {label} not found in map dict.")
            return f"OUT OF RANGE"

    def get_label_list(self):
        return self.labels

    def get_map_list(self):
        maplist = [self.label_to_gen[m] for m in self.labels]
        return maplist

    def gen_label_to_number(self, gen_label):
        if gen_label in results:
            return results[gen_label]
        else:
            return 3

    def print(self):
        print(f"Labels: {self.labels}")
        print(f"Map: {self.label_to_gen}")
