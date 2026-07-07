import cv2
import numpy as np
from typing import Optional

def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decodes raw image bytes into a 3-channel BGR OpenCV matrix.
    
    Args:
        image_bytes: Serialized image bytes (e.g. JPEG, PNG format).
        
    Returns:
        np.ndarray: BGR image matrix or None if decoding fails.
    """
    if not image_bytes:
        return None
    try:
        # Convert bytes to a 1D uint8 array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        # Avoid direct print, downstream caller handles logging or passes logger
        return None
