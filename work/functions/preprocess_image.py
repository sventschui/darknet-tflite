import cv2
import numpy as np
from typing import Tuple


def preprocess_image(
    input_width: int, input_height: int, image: np.ndarray
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess image for inference with padding

    Args:
        input_width: The expected input width
        input_height: The expected input height
        image: Input BGR image

    Returns:
        Preprocessed image, scale ratio, and padding offsets (pad_x, pad_y)
    """
    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate scaling ratio while maintaining aspect ratio
    scale = min(input_width / w, input_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded image (gray padding)
    padded = np.full((input_height, input_width, 3), 128, dtype=np.uint8)

    # Calculate padding offsets to center the image
    pad_x = (input_width - new_w) // 2
    pad_y = (input_height - new_h) // 2

    # Place resized image in center
    padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

    # Convert BGR to RGB and normalize to [0, 1]
    image_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0

    # Transpose to CHW format and add batch dimension
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_input = np.expand_dims(image_transposed, axis=0)

    return image_input, scale, (pad_x, pad_y)
