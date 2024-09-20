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

from typing import Tuple
import numpy as np
from postprocess import combined_nms, convert_to_ymin_xmin_ymax_xmax_format, BoxFormat, nms, scale_coords, clip_coords

def postprocess_yolov8_keypoints(outputs: Tuple[np.ndarray, np.ndarray, np.ndarray],
                                 conf: float = 0.3,
                                 iou_thres: float = 0.7,
                                 max_out_dets: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess the outputs of a YOLOv8 model for object detection and pose estimation.

    Args:
        outputs (Tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple containing the model outputs for bounding boxes,
            class predictions, and keypoint predictions.
        conf (float, optional): Confidence threshold for bounding box predictions. Default is 0.3
        iou_thres (float, optional): IoU (Intersection over Union) threshold for Non-Maximum Suppression (NMS).
            Default is 0.7.
        max_out_dets (int, optional): Maximum number of output detections to keep after NMS. Default is 300.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing the post-processed bounding boxes,
            their corresponding scores, and keypoints.

    """
    kpt_shape = (17, 3)
    feat_sizes = np.array([80, 40, 20])
    stride_sizes = np.array([8, 16, 32])
    a, s = (x.transpose() for x in make_anchors_yolo_v8(feat_sizes, stride_sizes, 0.5))

    y_bb, y_cls, kpts = outputs
    dbox = dist2bbox_yolo_v8(y_bb, a, xywh=True, dim=1) * s
    detect_out = np.concatenate((dbox, y_cls), 1)
    # additional part for pose estimation
    ndim = kpt_shape[1]
    pred_kpt = kpts.copy()
    if ndim == 3:
        pred_kpt[:, 2::3] = 1 / (1 + np.exp(-pred_kpt[:, 2::3]))  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
    pred_kpt[:, 0::ndim] = (pred_kpt[:, 0::ndim] * 2.0 + (a[0] - 0.5)) * s
    pred_kpt[:, 1::ndim] = (pred_kpt[:, 1::ndim] * 2.0 + (a[1] - 0.5)) * s

    x = np.concatenate([detect_out.transpose([2, 1, 0]).squeeze(), pred_kpt.transpose([2, 1, 0]).squeeze()], 1)
    x = x[(x[:, 4] > conf)]
    x = x[np.argsort(-x[:, 4])[:8400]]
    x[..., :4] = convert_to_ymin_xmin_ymax_xmax_format(x[..., :4], BoxFormat.XC_YC_W_H)
    boxes = x[..., :4]
    scores = x[..., 4]

    # Original post-processing part
    valid_indexs = nms(boxes, scores, iou_thres=iou_thres, max_out_dets=max_out_dets)
    x = x[valid_indexs]
    nms_bbox = x[:, :4]
    nms_scores = x[:, 4]
    nms_kpts = x[:, 5:]

    return nms_bbox, nms_scores, nms_kpts


def make_anchors_yolo_v8(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = np.arange(stop=w) + grid_cell_offset  # shift x
        sy = np.arange(stop=h) + grid_cell_offset  # shift y
        sy, sx = np.meshgrid(sy, sx, indexing='ij')
        anchor_points.append(np.stack((sx, sy), -1).reshape((-1, 2)))
        stride_tensor.append(np.full((h * w, 1), stride))
    return np.concatenate(anchor_points), np.concatenate(stride_tensor)


def dist2bbox_yolo_v8(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = np.split(distance, 2, axis=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return np.concatenate((c_xy, wh), dim)  # xywh bbox
    return np.concatenate((x1y1, x2y2), dim)  # xyxy bbox
