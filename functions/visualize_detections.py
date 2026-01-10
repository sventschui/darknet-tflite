import numpy as np
from typing import List
import cv2


def visualize_detections(
    image: np.ndarray, detections: List[dict], class_names: List[str]
) -> np.ndarray:
    """
    Visualize detections on image

    Args:
        image: Input BGR image
        detections: List of detections

    Returns:
        Image with visualized detections
    """
    vis_image = image.copy()

    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    colors[0] = (100, 100, 255)
    colors[1] = (0, 192, 0)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        class_id = det["class_id"]
        class_name = det["class_name"]

        # Get color for this class
        color = tuple(map(int, colors[class_id]))

        # Draw bounding box
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Prepare label
        label = f"{class_name}: {confidence:.2f}"

        # Get label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 0.8, 2
        )

        # Draw label background
        cv2.rectangle(
            vis_image, (x1, y1 - label_h - baseline - 8), (x1 + label_w, y1), color, -1
        )

        # Draw label text
        cv2.putText(
            vis_image,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_PLAIN,
            0.8,
            (255, 255, 255),
            2,
        )

    return vis_image
