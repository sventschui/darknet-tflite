import cv2
import numpy as np
from typing import Tuple, List


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iou_xyxy(box, boxes):
    # box: (4,), boxes: (N,4) with xyxy
    inter_x1 = np.maximum(box[0], boxes[:, 0])
    inter_y1 = np.maximum(box[1], boxes[:, 1])
    inter_x2 = np.minimum(box[2], boxes[:, 2])
    inter_y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-16
    return inter / union


def diou_xyxy(box, boxes):
    # DIoU metric: IoU - (rho^2 / c^2)
    iou = iou_xyxy(box, boxes)

    # centers
    bx = (box[0] + box[2]) * 0.5
    by = (box[1] + box[3]) * 0.5
    bxs = (boxes[:, 0] + boxes[:, 2]) * 0.5
    bys = (boxes[:, 1] + boxes[:, 3]) * 0.5
    rho2 = (bx - bxs) ** 2 + (by - bys) ** 2

    # enclosing box diagonal squared
    enc_x1 = np.minimum(box[0], boxes[:, 0])
    enc_y1 = np.minimum(box[1], boxes[:, 1])
    enc_x2 = np.maximum(box[2], boxes[:, 2])
    enc_y2 = np.maximum(box[3], boxes[:, 3])
    c2 = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-16

    return iou - (rho2 / c2)


def soft_nms_decay(iou_vals, sigma):
    # Gaussian Soft-NMS decay; score *= exp(- iou^2 / sigma)
    # (sigma ~ beta_nms). Clamp sigma to sane range
    sigma = max(1e-6, float(sigma))
    return np.exp(-(iou_vals**2) / sigma)


def decode_yolo_layer(
    layer_output: np.ndarray,
    layer_cfg: dict,
    input_w: int,
    input_h: int,
    conf_thresh: float,
    scale: float,
    padding: Tuple[int, int],
    class_names: List[str],
):
    """
    Decode a YOLO layer output to a list of detection dicts.

    Each detection dict has:
      {
        "bbox": [x1, y1, x2, y2],
        "confidence": float,
        "class_id": int,
        "class_name": str
      }

    Args:
        layer_output: np.ndarray, (B, A*(5+C), H, W)
        layer_cfg: dict, includes 'params' with anchors, mask, num, etc.
        input_w, input_h: model input size
        conf_thresh: minimum confidence to keep a detection
        scale, padding: preprocessing info (scale factor, (pad_x, pad_y))
        class_names: optional list of class names indexed by class_id
    """
    p = layer_cfg["params"]
    anchors_flat = p["anchors"]
    mask = p.get("mask", list(range(3)))
    scale_xy = float(p.get("scale_x_y", 1.0))
    pad_x, pad_y = padding

    B, ch, gh, gw = layer_output.shape
    na = len(mask)
    per_anchor = ch // na
    C = per_anchor - 5

    out = layer_output.reshape(B, na, 5 + C, gh, gw).transpose(0, 1, 3, 4, 2)
    tx = out[..., 0]
    ty = out[..., 1]
    tw = out[..., 2]
    th = out[..., 3]
    tobj = out[..., 4]
    tcls = out[..., 5:]

    all_anchors = np.array(anchors_flat, dtype=np.float32).reshape(-1, 2)
    anchors_wh = all_anchors[np.array(mask, dtype=np.int64)]

    # Grid
    grid_y, grid_x = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    grid_x = grid_x[None, None, ...].astype(np.float32)
    grid_y = grid_y[None, None, ...].astype(np.float32)

    # Decode (model input coords)
    cx = (sigmoid(tx) * scale_xy - 0.5 * (scale_xy - 1.0) + grid_x) / gw
    cy = (sigmoid(ty) * scale_xy - 0.5 * (scale_xy - 1.0) + grid_y) / gh
    pw = np.exp(tw) * anchors_wh[:, 0][None, :, None, None] / float(input_w)
    ph = np.exp(th) * anchors_wh[:, 1][None, :, None, None] / float(input_h)

    bx = cx * input_w
    by = cy * input_h
    bw = pw * input_w
    bh = ph * input_h

    x1 = bx - bw * 0.5
    y1 = by - bh * 0.5
    x2 = bx + bw * 0.5
    y2 = by + bh * 0.5

    obj = sigmoid(tobj)
    cls = sigmoid(tcls)
    scores = obj[..., None] * cls

    results = []
    for b in range(B):
        x1b = x1[b].reshape(-1)
        y1b = y1[b].reshape(-1)
        x2b = x2[b].reshape(-1)
        y2b = y2[b].reshape(-1)
        objb = obj[b].reshape(-1)
        scores_b = scores[b].reshape(-1, C)
        cls_ids = np.argmax(scores_b, axis=1)
        cls_scores = scores_b[np.arange(scores_b.shape[0]), cls_ids]

        keep = cls_scores >= conf_thresh
        if not np.any(keep):
            results.append([])
            continue

        dets = []
        for i in np.where(keep)[0]:
            dets.append(
                {
                    "bbox": [
                        float(x1b[i]),
                        float(y1b[i]),
                        float(x2b[i]),
                        float(y2b[i]),
                    ],
                    "confidence": float(cls_scores[i]),
                    "class_id": int(cls_ids[i]),
                    "class_name": (
                        class_names[cls_ids[i]]
                        if class_names and cls_ids[i] < len(class_names)
                        else f"class_{cls_ids[i]}"
                    ),
                    "objectness": float(objb[i]),
                }
            )
        results.append(dets)
    return results


def nms_per_class(
    detections, iou_thresh: float, kind="greedynms", beta_nms=0.6, max_det=300
):
    """
    Run NMS per class on detections in dict format.

    detections: list of dicts with keys:
        - 'bbox': [x1, y1, x2, y2]
        - 'confidence': float
        - 'class_id': int
        - 'class_name': str

    Returns: filtered list of detections (same format)
    """
    if detections is None or len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    class_ids = np.unique([d["class_id"] for d in detections])

    final_dets = []

    for cls_id in class_ids:
        cls_dets = [d for d in detections if d["class_id"] == cls_id]
        boxes = np.array([d["bbox"] for d in cls_dets], dtype=np.float32)
        scores = np.array([d["confidence"] for d in cls_dets], dtype=np.float32)

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0 and len(keep) < max_det:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            rest = order[1:]
            ious = iou_xyxy(boxes[i], boxes[rest])

            if kind == "diounms":
                metric = diou_xyxy(boxes[i], boxes[rest])
                suppress = np.where(metric > iou_thresh)[0]
                order = np.delete(order, suppress + 1)
                order = order[1:]
            elif kind == "soft":
                decay = soft_nms_decay(ious, sigma=beta_nms)
                scores[rest] *= decay
                valid = scores[rest] > 1e-6
                rest = rest[valid]
                order = np.concatenate(([i], rest[np.argsort(scores[rest])[::-1]]))
                order = order[1:]
            else:  # "greedynms"
                suppress = np.where(ious > iou_thresh)[0]
                order = np.delete(order, suppress + 1)
                order = order[1:]

        final_dets.extend([cls_dets[k] for k in keep])

    return final_dets


def apply_scale_and_padding(detections, scale, pad_x, pad_y):
    if detections is None or len(detections) == 0:
        return []

    adjusted = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # remove padding and scale
        x1 = round((x1 - pad_x) / scale)
        x2 = round((x2 - pad_x) / scale)
        y1 = round((y1 - pad_y) / scale)
        y2 = round((y2 - pad_y) / scale)
        det = det.copy()
        det["bbox"] = [x1, y1, x2, y2]
        adjusted.append(det)
    return adjusted


def postprocess_output(
    outputs: np.ndarray,
    yolo_layers_cfg: List[dict],
    input_size: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
    padding: Tuple[int, int],
    scale: float,
    class_names: List[str],
    max_det=300,
):
    """
    outputs: list[np.ndarray] or tuple[...] from the ONNX model,
                  one per YOLO layer, each of shape (B, A*(5+C), H, W).
    yolo_layers_cfg: list[dict], 1:1 with outputs, each like:
        {
          "type": "yolo",
          "params": {
             "anchors": [w1,h1,w2,h2,...],
             "mask": [m1,m2,m3],
             "num": 9,
             "scale_x_y": 1.05,
             "nms_kind": "greedynms" | "diounms" | "soft",
             "beta_nms": 0.6,
             "ignore_thresh": 0.7,
             "...": ...
          }
        }
    Returns:
        detections: list length B; each is (M,7) [x1,y1,x2,y2,score,class_id,obj] in input pixel coords
    """
    input_w, input_h = input_size
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    decoded_per_layer = []
    for out, cfg in zip(outputs, yolo_layers_cfg):
        decoded = decode_yolo_layer(
            out,
            cfg,
            input_w,
            input_h,
            conf_thresh=min(
                conf_threshold,
                float(cfg["params"].get("ignore_thresh", conf_threshold)),
            ),
            scale=scale,
            padding=padding,
            class_names=class_names,
        )
        decoded_per_layer.append(decoded)

    B = len(decoded_per_layer[0]) if decoded_per_layer else 0
    merged = []

    for b in range(B):
        all_dets = []
        for dl in decoded_per_layer:
            all_dets.extend(dl[b])

        if not all_dets:
            merged.append([])
            continue

        p0 = yolo_layers_cfg[0]["params"]
        nms_kind = str(p0.get("nms_kind", "greedynms")).lower()
        beta = float(p0.get("beta_nms", 0.6))

        kept = nms_per_class(
            all_dets,
            iou_thresh=iou_threshold,
            kind=nms_kind,
            beta_nms=beta,
            max_det=max_det,
        )

        if scale is not None and padding is not None:
            pad_x, pad_y = padding
            kept = apply_scale_and_padding(kept, scale, pad_x, pad_y)

        merged.append(kept)
    return merged


def postprocess_output_old(
    outputs: List[np.ndarray],
    scale: float,
    input_width: int,
    input_height: int,
    padding: Tuple[int, int],
    class_names: List[str],
    anchors: List[List[Tuple[int, int]]],
    conf_threshold: float,
    iou_threshold: float,
) -> List[dict]:
    """
    Postprocess YOLOv4 Darknet model outputs

    Args:
        outputs: Model outputs (list of detection layers)
        scale: Scale factor used in preprocessing
        padding: Padding offsets (pad_x, pad_y)

    Returns:
        List of detections with bbox, confidence, and class
    """
    pad_x, pad_y = padding
    boxes = []
    confidences = []
    class_ids = []
    num_classes = len(class_names)

    # YOLOv4 typically has 3 output layers for different scales
    # Map output to anchor index (13x13->0, 26x26->1, 52x52->2)
    for output_idx, output in enumerate(outputs):
        batch_size, num_channels, grid_h, grid_w = output.shape
        num_anchors = 3

        # Find which anchor set to use based on grid size
        anchor_idx = None
        if grid_h == 13:
            anchor_idx = 0
        elif grid_h == 26:
            anchor_idx = 1
        elif grid_h == 52:
            anchor_idx = 2

        if anchor_idx is None:
            continue

        anchors_for_scale = anchors[anchor_idx]

        # Reshape from (1, 255, H, W) to (1, 3, 85, H, W)
        output = output.reshape(
            batch_size, num_anchors, 5 + num_classes, grid_h, grid_w
        )

        # Transpose to (1, 3, H, W, 85)
        output = output.transpose(0, 1, 3, 4, 2)

        # Apply sigmoid activation selectively
        # Apply sigmoid to x, y (indices 0, 1), objectness (index 4), and class scores (5+)
        output[..., 0:2] = 1 / (1 + np.exp(-output[..., 0:2]))  # x, y
        output[..., 4:] = 1 / (1 + np.exp(-output[..., 4:]))  # objectness + classes

        # Process each anchor and grid cell
        for anchor_num in range(num_anchors):
            anchor_w, anchor_h = anchors_for_scale[anchor_num]

            for i in range(grid_h):
                for j in range(grid_w):
                    detection = output[0, anchor_num, i, j]

                    # detection format: [x, y, w, h, objectness, class_scores...]
                    objectness = detection[4]

                    if objectness > conf_threshold:
                        # Get class scores
                        class_scores = detection[5:]
                        class_id = np.argmax(class_scores)
                        class_confidence = class_scores[class_id]

                        # Combined confidence
                        confidence = objectness * class_confidence

                        if confidence > conf_threshold:
                            # Get box coordinates
                            box_x = detection[
                                0
                            ]  # offset in grid cell [0,1] after sigmoid
                            box_y = detection[
                                1
                            ]  # offset in grid cell [0,1] after sigmoid
                            box_w = detection[2]  # width (raw value)
                            box_h = detection[3]  # height (raw value)

                            # Convert to actual pixel coordinates in the input image
                            # Center coordinates
                            center_x = (j + box_x) * (input_width / grid_w)
                            center_y = (i + box_y) * (input_height / grid_h)

                            # Width and height using anchors
                            # YOLOv4: w = anchor_w * exp(tw), h = anchor_h * exp(th)
                            width = anchor_w * np.exp(box_w)
                            height = anchor_h * np.exp(box_h)

                            # Remove padding offset and scale back to original image
                            center_x = (center_x - pad_x) / scale
                            center_y = (center_y - pad_y) / scale
                            width = width / scale
                            height = height / scale

                            # Convert to corner coordinates
                            x1 = int(center_x - width / 2)
                            y1 = int(center_y - height / 2)

                            boxes.append([x1, y1, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(int(class_id))

    # Apply NMS
    if len(boxes) == 0:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

    detections = []
    for i in indices:
        box = boxes[i]
        x1, y1, w, h = box
        x2 = x1 + w
        y2 = y1 + h

        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "confidence": confidences[i],
                "class_id": class_ids[i],
                "class_name": class_names[class_ids[i]],
            }
        )

    return detections
