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

from abc import ABC, abstractmethod
from typing import Callable, Any


class AbstractWorker(ABC):
    def __init__(self):
        self.callback: Callable[[Any], None] = None

    @abstractmethod
    def do_work_callback(self, *args, **kwargs):
        """
        Perform work and call the callback function with the result.

        :param args: Positional arguments for the work.
        :param kwargs: Keyword arguments for the work.
        """
        pass
