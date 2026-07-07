import pytest
import numpy as np
from myface.matching.matcher import BiometricMatcher

def test_cosine_similarity():
    """Verifies cosine similarity boundaries."""
    matcher = BiometricMatcher()
    u = np.array([1.0, 2.0, 3.0])
    v = np.array([1.0, 2.0, 3.0])
    # Identical vectors should have cosine similarity of 1.0
    assert abs(matcher.compute_cosine_similarity(u, v) - 1.0) < 1e-5
    
    # Orthogonal vectors should have cosine similarity of 0.0
    w = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 1.0, 0.0])
    assert abs(matcher.compute_cosine_similarity(w, z) - 0.0) < 1e-5

def test_chi_square_distance():
    """Verifies Chi-Square distance on distribution vectors."""
    matcher = BiometricMatcher()
    p = np.array([0.1, 0.5, 0.4])
    q = np.array([0.1, 0.5, 0.4])
    # Identical distributions have 0.0 distance
    assert abs(matcher.compute_chi_square_distance(p, q) - 0.0) < 1e-9
    
    r = np.array([0.2, 0.4, 0.4])
    assert matcher.compute_chi_square_distance(p, r) > 0.0

def test_mahalanobis_distance():
    """Verifies Mahalanobis calculations for full and diagonal covariance matrices."""
    matcher = BiometricMatcher()
    query = np.array([1.0, 2.0])
    centroid = np.array([1.2, 1.8])
    
    # 1. Without covariance (falls back to Euclidean distance)
    dist_euclidean = matcher.compute_mahalanobis_distance(query, centroid)
    expected_euclidean = np.linalg.norm(query - centroid)
    assert abs(dist_euclidean - expected_euclidean) < 1e-5
    
    # 2. Diagonal covariance (variance vector)
    variances = np.array([0.1, 0.4])
    dist_diag = matcher.compute_mahalanobis_distance(query, centroid, variances)
    expected_diag = np.sqrt(((1.0 - 1.2)**2 / 0.1) + ((2.0 - 1.8)**2 / 0.4))
    assert abs(dist_diag - expected_diag) < 1e-5
    
    # 3. Full covariance matrix
    cov = np.array([[0.2, 0.1], [0.1, 0.3]])
    dist_full = matcher.compute_mahalanobis_distance(query, centroid, cov)
    assert dist_full > 0.0

def test_confidence_score_sigmoid():
    """Verifies sigmoid confidence bounds and thresholds."""
    matcher = BiometricMatcher()
    
    # Query matching threshold exactly has 50% confidence
    assert abs(matcher.compute_confidence_score(0.70, 0.70) - 0.50) < 1e-5
    
    # Score above threshold has high confidence
    assert matcher.compute_confidence_score(0.85, 0.70) > 0.90
    
    # Score below threshold has low confidence
    assert matcher.compute_confidence_score(0.55, 0.70) < 0.10

def test_threshold_calibration_far():
    """Verifies that FAR calibration correctly locates threshold index boundary."""
    matcher = BiometricMatcher()
    
    genuine_scores = [0.90, 0.92, 0.95]
    # Imposters scores sorted: [0.80, 0.75, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
    imposter_scores = [0.10, 0.75, 0.30, 0.60, 0.20, 0.80, 0.70, 0.50, 0.40, 0.05]
    
    # Desired FAR = 10% (0.10). Since we have 10 imposter scores,
    # the 10% FAR threshold should be the 1st elements (index 1 of sorted: 0.75)
    thresh = matcher.calibrate_threshold(genuine_scores, imposter_scores, target_far=0.10)
    assert abs(thresh - 0.75) < 1e-5

def test_verify_decision():
    """Verifies that biometric verify boundary decider operates correctly."""
    matcher = BiometricMatcher()
    u = np.array([1.0, 0.0])
    v = np.array([1.0, 0.0])
    
    # Perfect match (similarity score = 1.0) should pass threshold 0.8
    passed, score = matcher.verify(u, v, threshold=0.80)
    assert passed is True
    assert abs(score - 1.0) < 1e-5
