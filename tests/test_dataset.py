import os
import json
import pytest
import numpy as np
import cv2
from myface.dataset.manager import DatasetManager

def test_dataset_manager_initialization(test_data_dir):
    """Verifies directories are provisioned correctly on initialization."""
    mgr = DatasetManager(data_dir=test_data_dir)
    assert os.path.exists(mgr.raw_dir)
    assert os.path.exists(mgr.processed_dir)
    assert os.path.exists(mgr.rejected_dir)
    assert os.path.exists(mgr.enrollment_dir)
    assert os.path.exists(mgr.validation_dir)
    assert os.path.exists(mgr.spoof_dir)

def test_dataset_quality_metrics(test_data_dir):
    """Verifies sharpness, brightness, and contrast grading functions."""
    mgr = DatasetManager(data_dir=test_data_dir)
    
    # 1. Test clean sharp canvas (ideal metrics)
    img_clean = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_clean, (50, 50), 30, 255, -1) # Draw sharp edge
    q_score, sharpness, brightness, contrast = mgr.compute_quality_metrics(img_clean)
    assert sharpness > 0.0
    assert 0.0 <= q_score <= 1.0
    
    # 2. Test blurred canvas
    img_blurred = cv2.GaussianBlur(img_clean, (15, 15), 0)
    q_score_blur, sharpness_blur, _, _ = mgr.compute_quality_metrics(img_blurred)
    # Blurring must decrease sharpness variance
    assert sharpness_blur < sharpness
    assert q_score_blur < q_score

def test_duplicate_detection(test_data_dir):
    """Verifies histogram correlation and MAE duplicate detectors."""
    mgr = DatasetManager(data_dir=test_data_dir)
    
    img1 = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(img1, (20, 20), (80, 80), 255, -1)
    
    img2 = img1.copy()
    img3 = np.zeros((100, 100), dtype=np.uint8) # Distinct image
    
    existing = [img1]
    
    # Exact copy must be flagged as duplicate
    assert mgr.is_duplicate(img2, existing) is True
    # Distinct blank canvas must not be flagged
    assert mgr.is_duplicate(img3, existing) is False

def test_pose_estimation(test_data_dir):
    """Verifies that head pose profile estimation yields appropriate directions."""
    mgr = DatasetManager(data_dir=test_data_dir)
    
    # 1. Frontal (centered circle)
    img_front = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_front, (50, 50), 20, 255, -1)
    assert mgr.estimate_pose(img_front) == "Frontal"
    
    # 2. Right profile (offset right)
    img_right = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_right, (80, 50), 20, 255, -1)
    assert mgr.estimate_pose(img_right) == "Right Profile"
    
    # 3. Left profile (offset left)
    img_left = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img_left, (20, 50), 20, 255, -1)
    assert mgr.estimate_pose(img_left) == "Left Profile"

def test_batch_collection_workflow(test_data_dir):
    """Verifies that the collection pipeline filters frames and saves metadata JSON logs."""
    mgr = DatasetManager(data_dir=test_data_dir)
    user_id = "test_user_uuid"
    
    # Generate test snapshots: 1 sharp, 1 blurry (low sharpness), 1 duplicate
    img_good = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img_good, (50, 50), 30, (255, 255, 255), -1)
    
    # Blurry
    img_blurry = cv2.GaussianBlur(img_good, (19, 19), 0)
    
    # Duplicate
    img_duplicate = img_good.copy()
    
    frames = [img_good, img_blurry, img_duplicate]
    
    report = mgr.collect_enrollment_batch(
        user_id=user_id,
        frames=frames,
        blur_threshold=100.0, # Target sharpness rejection boundary
        quality_threshold=0.3,
        capture_interval=0.01
    )
    
    assert report["total_processed"] == 3
    assert report["total_accepted"] == 1 # First frame accepted
    assert report["total_rejected"] == 2 # Blurry and duplicate rejected
    
    # Verify metadata serialization
    accepted_filename = report["accepted_files"][0]
    accepted_meta_path = os.path.join(mgr.enrollment_dir, accepted_filename + ".json")
    assert os.path.exists(accepted_meta_path)
    
    with open(accepted_meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["user_id"] == user_id
    assert "image_quality" in metadata
    assert "pose_estimation" in metadata
