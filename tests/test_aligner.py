import pytest
import numpy as np
import cv2
from myface.alignment.aligner import FaceAligner
from myface.detection.detector import BBox

def test_aligner_initialization():
    """Verifies default values and parameter configurations."""
    aligner = FaceAligner(target_size=100, eye_distance_ratio=0.40)
    assert aligner.target_size == 100
    assert aligner.eye_distance_ratio == 0.40

def test_extract_eye_templates():
    """Verifies that pupil localizer extracts 24x24 eye template patches."""
    aligner = FaceAligner()
    
    # Create image with two dark circles representing eyes
    img = np.ones((100, 100), dtype=np.uint8) * 180
    # Left eye center (35, 40)
    cv2.circle(img, (35, 40), 0, 20, -1)
    # Right eye center (65, 40)
    cv2.circle(img, (65, 40), 0, 20, -1)
    
    bbox = BBox(x=10, y=10, w=80, h=80)
    
    templates = aligner.extract_eye_templates(img, bbox)
    assert "left_eye" in templates
    assert "right_eye" in templates
    assert templates["left_eye"].shape == (24, 24)
    assert templates["right_eye"].shape == (24, 24)
    # Verify templates are not empty and capture dark intensities
    assert np.mean(templates["left_eye"]) < 180

def test_locate_eyes():
    """Verifies template correlation matching identifies correct eye centers."""
    aligner = FaceAligner()
    
    # 1. Create a registration image and extract templates
    img_reg = np.ones((100, 100), dtype=np.uint8) * 180
    cv2.circle(img_reg, (35, 40), 0, 20, -1)
    cv2.circle(img_reg, (65, 40), 0, 20, -1)
    bbox = BBox(x=10, y=10, w=80, h=80)
    
    templates = aligner.extract_eye_templates(img_reg, bbox)
    
    # 2. Create a shifted query image
    img_query = np.ones((100, 100), dtype=np.uint8) * 180
    # Shift eyes slightly downwards/rightwards: Left (38, 43), Right (68, 43)
    cv2.circle(img_query, (38, 43), 0, 20, -1)
    cv2.circle(img_query, (68, 43), 0, 20, -1)
    
    left_center, right_center = aligner.locate_eyes(img_query, bbox, templates)
    
    # The located centers must match the shifted eye coordinates
    assert abs(left_center[0] - 38) <= 1
    assert abs(left_center[1] - 43) <= 1
    assert abs(right_center[0] - 68) <= 1
    assert abs(right_center[1] - 43) <= 1

def test_affine_alignment_rotation():
    """Verifies that affine warping corrects tilts and normalizes dimensions."""
    aligner = FaceAligner(target_size=128)
    
    # Create registration image to extract templates from
    img_reg = np.ones((120, 120), dtype=np.uint8) * 150
    cv2.circle(img_reg, (35, 40), 0, 20, -1)
    cv2.circle(img_reg, (65, 40), 0, 20, -1)
    bbox_reg = BBox(x=10, y=10, w=100, h=100)
    templates = aligner.extract_eye_templates(img_reg, bbox_reg)
    
    # Create image with tilted eyes: Left (30, 35), Right (70, 45)
    img = np.ones((120, 120), dtype=np.uint8) * 150
    cv2.circle(img, (30, 35), 0, 20, -1)
    cv2.circle(img, (70, 45), 0, 20, -1)
    bbox = BBox(x=10, y=10, w=100, h=100)
    
    aligned = aligner.align(img, bbox, templates)
    
    # Output must be normalized to target dimensions
    assert aligned.shape == (128, 128)
    
    # Since eyes are aligned horizontally, the pupil centers in the aligned image 
    # should be at y = 128 * 0.35 = 44.8, and symmetrically placed around x = 64.
    # Left eye: x = 64 - 128 * 0.35/2 = 41.6
    # Right eye: x = 64 + 128 * 0.35/2 = 86.4
    # The warped image will have dark spots around y=45, x=42 and x=86.
    assert aligned[45, 42] < 100
    assert aligned[45, 86] < 100
