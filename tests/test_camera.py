import cv2
import pytest
import time
import numpy as np
from unittest.mock import MagicMock, patch
from myface.camera.capture import CameraStream, CameraManager

@patch('myface.camera.capture.cv2.VideoCapture')
def test_camera_stream_lifecycle_and_acquisition(mock_video_capture):
    """
    Verifies that the CameraStream thread initializes, connects, reads frames, 
    populates timestamps, tracks health reports, and shuts down cleanly.
    """
    # Configure mock VideoCapture instance
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Simulate valid read frames
    dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    mock_cap.read.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: {
        3: 640.0, # CAP_PROP_FRAME_WIDTH
        4: 480.0  # CAP_PROP_FRAME_HEIGHT
    }.get(prop, 0.0)
    
    mock_video_capture.return_value = mock_cap

    # 1. Initialize and start stream
    stream = CameraStream(camera_idx=0, width=640, height=480, target_fps=30)
    assert stream.is_connected is False
    
    stream.start()
    time.sleep(0.1) # Yield time for loop read
    
    # 2. Assert acquisition states
    ret, frame, timestamp = stream.read_frame()
    assert ret is True
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert timestamp > 0.0
    
    # 3. Check health properties
    health = stream.get_health_status()
    assert health["camera_idx"] == 0
    assert health["is_connected"] is True
    assert health["total_frames_acquired"] > 0
    assert health["dropped_frames"] == 0
    assert health["resolution"] == "640x480"
    
    # 4. Terminate stream
    stream.stop()
    assert stream.is_connected is False
    assert stream.running is False

@patch('myface.camera.capture.cv2.VideoCapture')
def test_camera_manager_orchestration(mock_video_capture):
    """
    Verifies that the CameraManager scans active systems, provisions streams,
    re-configures active streams on parameter changes, and shuts down cleanly.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    
    props = {
        cv2.CAP_PROP_FRAME_WIDTH: 320.0,
        cv2.CAP_PROP_FRAME_HEIGHT: 240.0
    }
    def mock_set(prop, val):
        props[prop] = val
        return True
    mock_cap.set.side_effect = mock_set
    mock_cap.get.side_effect = lambda prop: props.get(prop, 0.0)
    
    mock_video_capture.return_value = mock_cap

    manager = CameraManager(default_width=320, default_height=240, default_fps=15)
    
    # 1. Test camera auto-detection
    active_indices = manager.detect_available_cameras(max_check=3)
    assert len(active_indices) == 3
    assert active_indices == [0, 1, 2]

    # 2. Get managed stream
    stream = manager.get_stream(camera_idx=0)
    assert stream.width == 320
    assert stream.height == 240
    assert stream.target_fps == 15
    time.sleep(0.05)
    
    # 3. Requesting same index with new configurations should update configuration parameters
    updated_stream = manager.get_stream(camera_idx=0, width=640, height=480, fps=30)
    assert updated_stream is stream # Same instance, updated params
    assert stream.width == 640
    assert stream.height == 480
    assert stream.target_fps == 30
    
    # 4. Clean shutdown releases all active managers
    manager.shutdown()
    assert stream.running is False
