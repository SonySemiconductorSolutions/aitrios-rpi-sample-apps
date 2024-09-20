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

from enum import Enum
from typing import List
import numpy as np
import cv2
import picamera2.sony_ivs as IVS
from picamera2 import Picamera2


def nms(dets: np.ndarray, scores: np.ndarray, iou_thres: float = 0.55, max_out_dets: int = 50) -> List[int]:
    """
    Perform Non-Maximum Suppression (NMS) on detected bounding boxes.

    Args:
        dets (np.ndarray): Array of bounding box coordinates of shape (N, 4) representing [y1, x1, y2, x2].
        scores (np.ndarray): Array of confidence scores associated with each bounding box.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.5.
        max_out_dets (int, optional): Maximum number of output detections to keep. Default is 300.

    Returns:
        List[int]: List of indices representing the indices of the bounding boxes to keep after NMS.

    """
    y1, x1 = dets[:, 0], dets[:, 1]
    y2, x2 = dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]

    return keep[:max_out_dets]


def combined_nms(batch_boxes, batch_scores, iou_thres: float = 0.65, conf: float = 0.55, max_out_dets: int = 50):
    nms_results = []
    for boxes, scores in zip(batch_boxes, batch_scores):
        xc = np.argmax(scores, 1)
        xs = np.amax(scores, 1)
        x = np.concatenate([boxes, np.expand_dims(xs, 1), np.expand_dims(xc, 1)], 1)

        xi = xs > conf
        x = x[xi]

        x = x[np.argsort(-x[:, 4])[:8400]]
        scores = x[:, 4]
        x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
        offset = x[:, 5] * 640
        boxes = x[..., :4] + np.expand_dims(offset, 1)

        # Original post-processing part
        valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
        x = x[valid_indexs]
        nms_classes = x[:, 5]
        nms_bbox = x[:, :4]
        nms_scores = x[:, 4]

        nms_results.append((nms_bbox, nms_scores, nms_classes))

    return nms_results


class BoxFormat(Enum):
    YMIM_XMIN_YMAX_XMAX = 'ymin_xmin_ymax_xmax'
    XMIM_YMIN_XMAX_YMAX = 'xmin_ymin_xmax_ymax'
    XMIN_YMIN_W_H = 'xmin_ymin_width_height'
    XC_YC_W_H = 'xc_yc_width_height'


def convert_to_ymin_xmin_ymax_xmax_format(boxes, orig_format: BoxFormat):
    """
    changes the box from one format to another (XMIN_YMIN_W_H --> YMIM_XMIN_YMAX_XMAX )
    also support in same format mode (returns the same format)

    :param boxes:
    :param orig_format:
    :return: box in format YMIM_XMIN_YMAX_XMAX
    """
    if len(boxes) == 0:
        return boxes
    elif orig_format == BoxFormat.YMIM_XMIN_YMAX_XMAX:
        return boxes
    elif orig_format == BoxFormat.XMIN_YMIN_W_H:
        boxes[:, 2] += boxes[:, 0]  # convert width to xmax
        boxes[:, 3] += boxes[:, 1]  # convert height to ymax
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XMIM_YMIN_XMAX_YMAX:
        boxes[:, 0], boxes[:, 1] = boxes[:, 1], boxes[:, 0].copy()  # swap xmin, ymin columns
        boxes[:, 2], boxes[:, 3] = boxes[:, 3], boxes[:, 2].copy()  # swap xmax, ymax columns
        return boxes
    elif orig_format == BoxFormat.XC_YC_W_H:
        new_boxes = np.copy(boxes)
        new_boxes[:, 0] = boxes[:, 1] - boxes[:, 3] / 2  # top left y
        new_boxes[:, 1] = boxes[:, 0] - boxes[:, 2] / 2  # top left x
        new_boxes[:, 2] = boxes[:, 1] + boxes[:, 3] / 2  # bottom right y
        new_boxes[:, 3] = boxes[:, 0] + boxes[:, 2] / 2  # bottom right x
        return new_boxes
    else:
        raise Exception("Unsupported boxes format")


def clip_boxes(boxes: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip bounding boxes to stay within the image boundaries.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    boxes[..., 0] = np.clip(boxes[..., 0], a_min=0, a_max=h)
    boxes[..., 1] = np.clip(boxes[..., 1], a_min=0, a_max=w)
    boxes[..., 2] = np.clip(boxes[..., 2], a_min=0, a_max=h)
    boxes[..., 3] = np.clip(boxes[..., 3], a_min=0, a_max=w)
    return boxes


def scale_boxes(boxes: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int, preserve_aspect_ratio: bool,
                normalized: bool = True) -> np.ndarray:
    """
    Scale and offset bounding boxes based on model output size and original image size.

    Args:
        boxes (numpy.ndarray): Array of bounding boxes in format [y_min, x_min, y_max, x_max].
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    nh, nw = (H, W) if normalized else (1, 1)

    # Scale and offset boxes
    # [y_min, x_min, y_max, x_max].
    boxes[..., 0] = (boxes[..., 0] * nw - deltaW) * scale_W
    boxes[..., 1] = (boxes[..., 1] * nh - deltaH) * scale_H
    boxes[..., 2] = (boxes[..., 2] * nw - deltaW) * scale_W
    boxes[..., 3] = (boxes[..., 3] * nh - deltaH) * scale_H

    # Clip boxes
    boxes = clip_boxes(boxes, h_image, w_image)

    return boxes


def scale_coords(kpts: np.ndarray, h_image: int, w_image: int, h_model: int, w_model: int,
                 preserve_aspect_ratio: bool) -> np.ndarray:
    """
    Scale and offset keypoints based on model output size and original image size.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h_image (int): Original image height.
        w_image (int): Original image width.
        h_model (int): Model output height.
        w_model (int): Model output width.
        preserve_aspect_ratio (bool): Whether to preserve image aspect ratio during scaling

    Returns:
        numpy.ndarray: Scaled and offset bounding boxes.
    """
    deltaH, deltaW = 0, 0
    H, W = h_model, w_model
    scale_H, scale_W = h_image / H, w_image / W

    if preserve_aspect_ratio:
        scale_H = scale_W = max(h_image / H, w_image / W)
        H_tag = int(np.round(h_image / scale_H))
        W_tag = int(np.round(w_image / scale_W))
        deltaH, deltaW = int((H - H_tag) / 2), int((W - W_tag) / 2)

    # Scale and offset boxes
    kpts[..., 0] = (kpts[..., 0] - deltaH) * scale_H
    kpts[..., 1] = (kpts[..., 1] - deltaW) * scale_W

    # Clip boxes
    kpts = clip_coords(kpts, h_image, w_image)

    return kpts


def clip_coords(kpts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Clip keypoints to stay within the image boundaries.

    Args:
        kpts (numpy.ndarray): Array of bounding keypoints in format [..., 17, 3]  where the last dim is (x, y, visible).
        h (int): Height of the image.
        w (int): Width of the image.

    Returns:
        numpy.ndarray: Clipped bounding boxes.
    """
    kpts[..., 0] = np.clip(kpts[..., 0], a_min=0, a_max=h)
    kpts[..., 1] = np.clip(kpts[..., 1], a_min=0, a_max=w)
    return kpts


PARTS = {
    0: 'Nose',
    1: 'EyeL',
    2: 'EyeR',
    3: 'EarL',
    4: 'EarR',
    5: 'SholderL',
    6: 'SholderR',
    7: 'ElbowL',
    8: 'ElbowR',
    9: 'WristL',
    10: 'WristR',
    11: 'HipL',
    12: 'HipR',
    13: 'KneeL',
    14: 'KneeR',
    15: 'AnkleL',
    16: 'AnkleR'
}


class COCODrawer:
    def __init__(self, categories, imx500: IVS.ivs):
        self.categories = categories
        self.imx500 = imx500

    def draw_bounding_box(self, img, annotation, class_id, score,   metadata: dict, picam2: Picamera2,  stream):
        obj_scaled = self.imx500.convert_inference_coords(annotation, metadata, picam2, stream)
        x_min = obj_scaled.x
        y_min = obj_scaled.y
        x_max = x_min + obj_scaled.width
        y_max = y_min + obj_scaled.height
        text = f"{self.categories[int(class_id)]}:{score:.3f}"
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(img, text, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    def draw_keypoints(self, img, keypoints, min_confidence,  metadata: dict, picam2: Picamera2, stream):
        def get_point(index):
            y0, x0 = keypoints[index][1], keypoints[index][0]
            coords = y0, x0, y0 + 1, x0 + 1
            obj_scaled = self.imx500.convert_inference_coords(coords, metadata, picam2, stream)
            return obj_scaled.x, obj_scaled.y

        skeleton = [
            [0, 1], [0, 2], [1, 3], [2, 4],  # Head
            [5, 6], [5, 7], [7, 9], [6, 8],  # Arms
            [8, 10], [5, 11], [6, 12], [11, 12],  # Body
            [11, 13], [12, 14], [13, 15], [14, 16]  # Legs
        ]

        # Draw skeleton lines
        for connection in skeleton:
            start_point = get_point(connection[0])
            end_point = get_point(connection[1])
            start_confidence = keypoints[connection[0]][2]
            end_confidence = keypoints[connection[1]][2]
            if start_confidence < min_confidence or end_confidence < min_confidence:
                continue
            cv2.line(img, start_point, end_point, (255, 0, 0), 2)

        # Draw keypoints as colored circles
        for i in range(len(keypoints)):
            x, y = get_point(i)
            confidence = keypoints[i][2]
            if confidence < min_confidence:
                continue
            cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            label = f"{PARTS.get(i)}.{confidence:.3f}"
            cv2.putText(img, label, (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 0), 1)

    def annotate_image(self, img, b, s, c, k, box_min_conf, kps_min_conf,  metadata: dict, picam2: Picamera2,  stream):
        for index, row in enumerate(b):
            if s[index] >= box_min_conf:
                self.draw_bounding_box(img, row, c[index], s[index],  metadata, picam2, stream)
                if k is not None:
                    self.draw_keypoints(img, k[index], kps_min_conf,  metadata, picam2, stream)


def softmax(x):
    y = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    z = y / np.expand_dims(np.sum(y, axis=-1), axis=-1)
    return z
