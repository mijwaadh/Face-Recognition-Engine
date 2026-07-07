import pytest
import numpy as np
import cv2
from myface.quality_assessment.assessment import FrameQualityEvaluator
from myface.detection.detector import BBox

def test_image_quality_metrics():
    """Verifies blur, focus, brightness, contrast, noise, and exposure calculations."""
    evaluator = FrameQualityEvaluator(
        min_blur_threshold=50.0,
        min_focus_threshold=200.0,
        min_contrast_threshold=10.0,
        min_exposure_threshold=0.80,
        max_noise_threshold=15.0
    )
    
    # 1. Clean synthetic sharp canvas (should pass)
    img_clean = np.ones((100, 100), dtype=np.uint8) * 100
    cv2.circle(img_clean, (50, 50), 30, 180, -1)
    
    report_clean = evaluator.evaluate_frame(img_clean)
    assert report_clean.blur_score > 0.0
    assert report_clean.focus_score > 0.0
    assert report_clean.passed is True
    assert len(report_clean.rejection_reasons) == 0
    
    # 2. Blurred canvas (should fail blur and focus checks)
    img_blurred = cv2.GaussianBlur(img_clean, (19, 19), 0)
    report_blurred = evaluator.evaluate_frame(img_blurred)
    assert report_blurred.passed is False
    assert any("blur" in r or "focus" in r for r in report_blurred.rejection_reasons)

def test_face_quality_metrics():
    """Verifies face size, pose yaw asymmetry, and spatial block-based occlusion calculations."""
    evaluator = FrameQualityEvaluator(min_contrast_threshold=20.0)
    
    # Face crop dimensions
    img = np.ones((200, 200, 3), dtype=np.uint8) * 100
    
    # 1. Good frontal face box placement
    # Draw simple shapes to simulate face components
    cv2.circle(img, (100, 100), 40, (180, 180, 180), -1)
    bbox_good = BBox(x=60, y=60, w=80, h=80)
    
    report_good = evaluator.evaluate_frame(img, bbox_good)
    assert report_good.face_size_score is not None
    assert report_good.pose_score is not None
    assert report_good.occlusion_score is not None
    assert report_good.passed is True
    
    # 2. Too small face size ratio
    bbox_small = BBox(x=95, y=95, w=10, h=10)
    report_small = evaluator.evaluate_frame(img, bbox_small)
    assert report_small.passed is False
    assert any("size" in r for r in report_small.rejection_reasons)
    
    # 3. High asymmetry (pose yaw)
    evaluator_profile = FrameQualityEvaluator(min_exposure_threshold=0.0)
    img_profile = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img_profile, (140, 100), 40, (255, 255, 255), -1) # Offset circle
    bbox_profile = BBox(x=60, y=60, w=80, h=80)
    report_profile = evaluator_profile.evaluate_frame(img_profile, bbox_profile)
    assert report_profile.passed is False
    assert any("pose" in r or "overall" in r for r in report_profile.rejection_reasons)

def test_occlusion_estimation():
    """Verifies flat zone block analysis filters face occlusions."""
    evaluator = FrameQualityEvaluator()
    
    # Clean non-occluded canvas
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Fill with random texture to avoid low variance checks
    np.random.seed(42)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    bbox = BBox(x=0, y=0, w=100, h=100)
    
    report_clean = evaluator.evaluate_frame(img, bbox)
    assert report_clean.occlusion_score > 0.9
    
    # Occlude half of the canvas (all constant zero pixels)
    img_occluded = img.copy()
    img_occluded[0:50, 0:100] = 0 # Occlude top half with black block
    
    report_occluded = evaluator.evaluate_frame(img_occluded, bbox)
    # The top half has 8 blocks of all zeros (std = 0.0), so occlusion_score drops
    assert report_occluded.occlusion_score < 0.6
    assert report_occluded.passed is False
    assert any("occlusion" in r for r in report_occluded.rejection_reasons)
