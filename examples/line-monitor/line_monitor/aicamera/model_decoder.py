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

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import Callable, Any


class BaseTensorDecoder(ABC):
    @abstractmethod
    def decode(self, tensor):
        """
        Decodes the given tensor into a human-readable or structured format.

        :param tensor: The output tensor from the AI model.
        :return: Decoded result.
        """
        pass

    @abstractmethod
    def preprocess(self, tensor):
        """
        Optionally preprocess the tensor before decoding.

        :param tensor: The output tensor from the AI model.
        :return: Preprocessed tensor.
        """
        pass

    @abstractmethod
    def postprocess(self, decoded_output):
        """
        Optionally postprocess the decoded output.

        :param decoded_output: The result from the decode method.
        :return: Postprocessed result.
        """
        pass


class ClassificationDecoder(BaseTensorDecoder):
    def __init__(self, labels_filename="labels.txt"):
        self.labels = []
        print(f"Will use {labels_filename}.")
        with open(labels_filename, "r") as file:
            lines = file.readlines()
            self.labels = [line.strip() for line in lines]
        print(f"Labels: {self.labels}")

    def decode(self, tensor):
        """
        Decodes the classification tensor to a class label.

        :param tensor: The output tensor from the AI model, typically a list of probabilities.
        :return: The predicted class label.
        """

        class_index = np.argmax(tensor)
        return class_index

    def preprocess(self, tensor):
        """
        Example of preprocessing, such as normalization.

        :param tensor: The output tensor from the AI model.
        :return: Preprocessed tensor.
        """

        return tensor

    def postprocess(self, decoded_output):
        """
        Example of postprocessing, such as mapping index to class label.

        :param decoded_output: The result from the decode method.
        :return: Human-readable class label.
        """
        class_labels = self.labels
        return class_labels[decoded_output]

    def get_prediction_details(self, tensor, timestamp):
        """
        Returns the index, label, and score as a dictionary.

        :param tensor: The output tensor from the AI model, typically a list of probabilities.
        :return: A dictionary containing the index, label, and score.
        """
        class_index = self.decode(tensor)
        label = self.postprocess(class_index)
        score = tensor[class_index]
        return {"C": int(class_index), "label": label, "P": score, "T": timestamp}
