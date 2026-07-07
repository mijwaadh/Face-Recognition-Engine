import pytest
import numpy as np
import cv2
from myface.detection.detector import FaceDetector, BBox

def test_face_detector_initialization():
    """Verifies detector initializes parameters and synthetic template size."""
    detector = FaceDetector(min_size=60, confidence_threshold=0.50)
    assert detector.min_size == 60
    assert detector.face_template_hog is not None
    # HOG feature vector length: 7x7 cells * 4 blocks * 9 bins = 1764 elements
    assert len(detector.face_template_hog) == 1764

def test_symmetry_score_mapping():
    """Verifies that symmetry axis calculations peak at the geometric center."""
    detector = FaceDetector()
    
    # Create horizontally symmetric canvas: vertical bar in center
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (50, 10), (50, 90), 255, 6) # Draw center axis line
    
    # Smooth to generate gradient spreads
    img_smooth = cv2.GaussianBlur(img, (5, 5), 0)
    
    scores = detector.compute_symmetry_map(img_smooth, scale_w=40)
    # Symmetry scores must peak around center column c=50
    center_score = scores[50]
    side_score = scores[20]
    
    assert center_score > 0.4
    assert center_score > side_score

def test_projection_profile():
    """Verifies that structural horizontal bands pass profile validations."""
    detector = FaceDetector()
    
    # 1. Structured face mockup (horizontal stripes modeling eyes/mouth)
    face_crop = np.zeros((84, 60), dtype=np.uint8)
    # Eyes band (upper half)
    cv2.line(face_crop, (5, 20), (55, 20), 255, 3)
    # Mouth band (lower half)
    cv2.line(face_crop, (15, 60), (45, 60), 255, 3)
    
    # Smooth
    face_crop = cv2.GaussianBlur(face_crop, (5, 5), 0)
    
    score = detector.verify_projection_profile(face_crop)
    # Must pass (score > 0) due to eyes/mouth peaks and nose bridge valley
    assert score > 0.3

def test_non_maximum_suppression():
    """Verifies NMS keeps the best candidate and suppresses overlapping boxes."""
    detector = FaceDetector(nms_iou_threshold=0.3)
    
    box1 = BBox(x=10, y=10, w=100, h=140)
    box2 = BBox(x=12, y=12, w=100, h=140) # Highly overlapping (IoU > 0.8)
    box3 = BBox(x=200, y=10, w=100, h=140) # Separate distinct box (IoU = 0.0)
    
    candidates = [
        (box1, 0.90),
        (box2, 0.80),
        (box3, 0.85)
    ]
    
    results = detector._compute_nms_candidates(candidates)
    assert len(results) == 2
    # Check that best box is preserved and overlap suppressed
    assert results[0][0] == box1
    assert results[1][0] == box3

def test_full_detection_on_synthetic_face():
    """Verifies that the detector identifies a programmatically drawn face canvas."""
    detector = FaceDetector(min_size=60, confidence_threshold=0.55)
    
    # Create canvas size 160x160
    img = np.ones((160, 160, 3), dtype=np.uint8) * 100
    
    # Draw simple face shape centered at (80, 80)
    # Width 80, Height 112 (aspect ratio 1.4)
    cv2.ellipse(img, (80, 80), (25, 35), 0, 0, 360, (180, 180, 180), -1) # Head
    cv2.line(img, (65, 70), (75, 70), (80, 80, 80), 2)                  # Left eye
    cv2.line(img, (85, 70), (95, 70), (80, 80, 80), 2)                  # Right eye
    cv2.line(img, (80, 72), (80, 88), (110, 110, 110), 2)               # Nose
    cv2.line(img, (70, 95), (90, 95), (60, 60, 60), 2)                  # Mouth
    
    img_smooth = cv2.GaussianBlur(img, (5, 5), 0)
    
    bbox = detector.detect(img_smooth)
    assert bbox is not None
    # BBox center should be close to (80, 80)
    center_x = bbox.x + bbox.w // 2
    center_y = bbox.y + bbox.h // 2
    assert abs(center_x - 80) < 15
    assert abs(center_y - 80) < 20
