"""
Camera sub-package.
Provides high-performance multi-threaded frame acquisition classes.
"""

from myface.camera.capture import CameraStream, CameraManager

__all__ = ["CameraStream", "CameraManager"]
