import pytest
import numpy as np
import cv2
from myface.anti_spoof.liveness import LivenessDetector

def test_liveness_detector_initialization():
    """Verifies default parameters and weight vectors."""
    detector = LivenessDetector(weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], bias=-2.0)
    assert len(detector.weights) == 6
    assert detector.bias == -2.0

def test_sfs_depth_variance():
    """Verifies that 3D curved surfaces yield higher depth variance than flat regions."""
    detector = LivenessDetector()
    
    # 1. Flat canvas (spoof)
    img_flat = np.ones((64, 64), dtype=np.uint8) * 128
    score_flat = detector.compute_sfs_depth_variance(img_flat)
    
    # 2. Curved sphere-like canvas (live)
    img_curved = np.zeros((64, 64), dtype=np.uint8)
    cv2.circle(img_curved, (32, 32), 20, 200, -1)
    img_curved = cv2.GaussianBlur(img_curved, (15, 15), 0)
    score_curved = detector.compute_sfs_depth_variance(img_curved)
    
    assert score_curved > score_flat

def test_moire_pattern_peaks():
    """Verifies that high-frequency screen noise patterns generate Moire peaks."""
    detector = LivenessDetector()
    
    # 1. Clean grayscale canvas
    img_clean = np.ones((64, 64), dtype=np.uint8) * 128
    score_clean = detector.detect_moire_fft_peaks(img_clean)
    assert score_clean == 0.0
    
    # 2. Canvas with simulated high-frequency screen grids
    img_grid = np.ones((64, 64), dtype=np.uint8) * 128
    # Draw horizontal stripes
    img_grid[::4, :] = 200
    # Draw vertical stripes
    img_grid[:, ::4] = 200
    
    score_grid = detector.detect_moire_fft_peaks(img_grid)
    assert score_grid > 0.0

def test_chrominance_diffusion():
    """Verifies skin-tone chrominance diffusion variances."""
    detector = LivenessDetector()
    
    # 1. Smooth chrominance image (live skin)
    img_smooth = np.zeros((64, 64, 3), dtype=np.uint8)
    img_smooth[:, :, 0] = 100 # B
    img_smooth[:, :, 1] = 110 # G
    img_smooth[:, :, 2] = 180 # R (Reddish skin-like)
    score_smooth = detector.compute_chrominance_diffusion(img_smooth)
    
    # 2. Noisy print-tone image (halftones/sharp margins)
    np.random.seed(42)
    img_noisy = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    score_noisy = detector.compute_chrominance_diffusion(img_noisy)
    
    assert score_smooth > score_noisy

def test_lbp_texture_analysis():
    """Verifies micro-texture variance maps."""
    detector = LivenessDetector()
    
    img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    score = detector.analyze_lbp_texture(img)
    assert 0.0 <= score <= 1.0

def test_optical_flow_magnitude():
    """Verifies dense flow rigidity analysis comparing rigid translate vs non-rigid motion."""
    detector = LivenessDetector()
    
    # 1. Uniform translation (rigid movement)
    frame1 = np.ones((64, 64), dtype=np.uint8) * 128
    cv2.circle(frame1, (32, 32), 10, 200, -1)
    
    # Shift circle slightly to left
    frame2 = np.ones((64, 64), dtype=np.uint8) * 128
    cv2.circle(frame2, (30, 32), 10, 200, -1)
    
    score_rigid = detector.compute_optical_flow_magnitude(frame1, frame2)
    
    # 2. Static frames transition
    score_static = detector.compute_optical_flow_magnitude(frame1, frame1)
    
    assert score_rigid > 0.0
    assert score_static == 0.2

def test_logistic_sigmoid_fusion():
    """Verifies classification output probability mapping."""
    detector = LivenessDetector(weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0], bias=-4.0)
    
    # 1. High scores on liveness markers (should pass liveness)
    # SfS=0.9, Moire=0.0 (inverted 1.0), Diff=0.9, LBP=0.8, Sharp=0.9, Flow=0.8
    # Logit = 2*0.9 + 2*1.0 + 1*0.9 + 1*0.8 + 1*0.9 + 1*0.8 - 4.0 = 1.8 + 2.0 + 0.9 + 0.8 + 0.9 + 0.8 - 4.0 = 7.2 - 4.0 = 3.2
    # Sigmoid(3.2) = 0.96
    img_live = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(img_live, (32, 32), 15, (100, 110, 180), -1)
    img_live = cv2.GaussianBlur(img_live, (5, 5), 0)
    
    is_live, prob = detector.analyze_liveness(img_live, threshold=0.70)
    assert prob > 0.50
