import numpy as np
from typing import Dict, Tuple, List
import onnxruntime as ort
import cv2

def _apply_nms(detections: List[Dict], iou_threshold: float, conf_threshold: float) -> List[Dict]:
    """
    Apply Non-Maximum Suppression (NMS) using OpenCV dnn.NMSBoxes per class.
    """
    if len(detections) == 0:
        return []

    results = []
    class_ids = set(d['class_id'] for d in detections)

    for cls in class_ids:
        cls_dets = [d for d in detections if d['class_id'] == cls]
        boxes = []
        scores = []

        for d in cls_dets:
            x1, y1, x2, y2 = d['bbox']
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(d['confidence'])

        indices = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, iou_threshold)
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(cls_dets[i])

    return results

def parse_onnx_outputs(
    session: ort.InferenceSession,
    outputs: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ONNX outputs with strict name checking.

    Expected ONNX graph outputs:
        - "confs"
        - "boxes"

    Raises:
        RuntimeError if outputs are missing or misnamed.
    """

    output_names = [o.name for o in session.get_outputs()]

    try:
        conf_idx = output_names.index("confs")
        box_idx = output_names.index("boxes")
    except ValueError:
        raise RuntimeError(
            f"Expected ONNX outputs named 'confs' and 'boxes', "
            f"but found: {output_names}"
        )

    return (
        np.asarray(outputs[conf_idx]),
        np.asarray(outputs[box_idx]),
    )

def postprocess_output(
    confs: np.ndarray,
    bboxes: np.ndarray,
    class_names: List[str],
    image_size: Tuple[int, int],
    scale: float,
    padding: Tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> List[List[Dict]]:
    """
    Post-process Darknet v5.1 ONNX outputs for a batch, applying letterbox undo.

    Args:
        confs: np.ndarray of shape (B, N, num_classes)
        bboxes: np.ndarray of shape (B, N, 1, 4) (normalized or absolute in letterbox space)
        class_names: list of class names
        image_size: (width, height) of the original image
        scale: letterbox scaling factor used during preprocessing
        padding: (pad_x, pad_y) applied during letterbox preprocessing
        conf_threshold: minimum confidence to keep a detection
        iou_threshold: IoU threshold for NMS

    Returns:
        List of lists of dicts, one per batch image:
        [
            [
                {"bbox": [x1, y1, x2, y2], "class_id": int, "class_name": str, "confidence": float},
                ...
            ],
            ...
        ]
    """

    image_width, image_height = image_size
    pad_x, pad_y = padding
    batch_size = confs.shape[0]
    results_batch: List[List[Dict]] = []

    for batch_idx in range(batch_size):
        confs_img = confs[batch_idx]   # (N, num_classes)
        bboxes_img = bboxes[batch_idx] # (N, 1, 4)

        # Remove singleton dimension
        bboxes_img = np.squeeze(bboxes_img, axis=1)  # (N, 4)

        # Best class per box
        class_ids = np.argmax(confs_img, axis=1)
        scores = confs_img[np.arange(len(confs_img)), class_ids]

        results_img: List[Dict] = []

        for bbox, cls_id, score in zip(bboxes_img, class_ids, scores):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = bbox

            # If boxes are normalized, scale to letterboxed input size
            x1 *= image_width
            x2 *= image_width
            y1 *= image_height
            y2 *= image_height

            # Undo letterbox: remove padding and divide by scale
            x1 = (x1 - pad_x) / scale
            x2 = (x2 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            y2 = (y2 - pad_y) / scale
            

            results_img.append({
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "class_id": int(cls_id),
                "class_name": class_names[cls_id],
                "confidence": float(score),
            })

        results_img = _apply_nms(results_img, iou_threshold=iou_threshold, conf_threshold=conf_threshold)

        results_batch.append(results_img)

    return results_batch