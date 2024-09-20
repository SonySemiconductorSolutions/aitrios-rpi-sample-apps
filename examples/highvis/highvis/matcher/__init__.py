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

from .overlay_detector import OverlapDetector
from .object_event import ObjectEventLogic

from .tracker_helper import *
from .example.utils import settings, shutdown_handler
from .example.annotation_work_item import WorkQueueAnnotator


an_colors = [["#ff00ff", "#002200", "#000066"], ["#ff00ff", "#002200", "#000066"]]
