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

import json
import os


class JSONFileManager:
    def __init__(self, default_data, file_path="settings.json"):
        self.labels = []
        self.json_data = default_data

        self.file_path = file_path

        if not self.file_exists(self.file_path):
            self.create_default_json_file()
            exit(1)

        self.read_json_file()

    def file_exists(self, file_path):
        return os.path.exists(file_path)

    def read_json_file(self, file_path=None):
        if file_path is None:
            file_path = self.file_path

        if file_path is None:
            raise ValueError("No file path specified.")

        with open(file_path, "r") as file:
            self.json_data = json.load(file)

        return self.json_data

    def save_json_file(self, file_path=None, json_data=None):
        if file_path is None:
            file_path = self.file_path

        if file_path is None:
            raise ValueError("No file path specified.")

        if json_data is None:
            json_data = self.json_data

        with open(file_path, "w") as file:
            json.dump(json_data, file, indent=4)

    def create_default_json_file(self, file_path=None):
        if file_path is None:
            file_path = self.file_path

        if file_path is None:
            raise ValueError("No file path specified.")

        if not self.file_exists(file_path):
            self.save_json_file(file_path, self.json_data)
            print(f"Default JSON file created: {file_path}")

    def read_labels_file(self, file_path="labels.txt"):
        self.labels = []
        if self.file_exists(file_path):
            print(f"Reading {file_path}")
            with open(file_path, "r") as file:
                for line in file:
                    self.labels.append(line.strip().upper())
                    print(f"{line.strip()}")
            return True
        else:
            print(f"File {file_path} doesn't exist.")
            return False

    def get_labels(self):
        for idx, label in enumerate(self.labels, 0):
            print(f"{idx}. {label}")
        return self.labels

    def get_json_data(self):
        print(json.dumps(self.json_data, indent=2))
        return self.json_data
