import pytest
import numpy as np
from myface.feature_extraction.extractor import FeatureExtractor

def test_feature_extractor_initialization():
    """Verifies default values and parameter configurations."""
    extractor = FeatureExtractor(lbp_grid=(4, 4), hog_bins=8)
    assert extractor.lbp_grid == (4, 4)
    assert extractor.hog_bins == 8

def test_lbp_histogram_shapes():
    """Verifies LBP shape outputs match spatial grid partitions."""
    extractor = FeatureExtractor(lbp_grid=(8, 8))
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    lbp_feat = extractor.extract_lbp_histogram(img)
    # 8x8 grid cells * 256 bins per cell = 16384 features
    assert lbp_feat.shape == (16384,)
    # Verify histograms are normalized (should sum to close to cell counts)
    assert abs(np.linalg.norm(lbp_feat[:256]) - 1.0) < 1e-4

def test_hog_descriptor_shapes():
    """Verifies HOG shape outputs match overlap block calculations."""
    extractor = FeatureExtractor(hog_bins=9)
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    hog_feat = extractor.extract_hog_descriptor(img)
    # 128x128 image has 8x8 cells of size 16x16
    # 2x2 cells overlapping blocks with 1 overlap cell = 7x7 = 49 blocks
    # 49 blocks * 4 cells/block * 9 bins/cell = 1764 features
    assert hog_feat.shape == (1764,)

def test_feature_fusion_and_normalization():
    """Verifies concatenated descriptor sizes and unit vector scaling."""
    extractor = FeatureExtractor()
    img = np.ones((128, 128), dtype=np.uint8) * 128
    
    combined = extractor.extract_features(img)
    # LBP (16384) + HOG (1764) = 18148 features
    assert combined.shape == (18148,)
    # Fuzed vector must be normalized to unit length
    assert abs(np.linalg.norm(combined) - 1.0) < 1e-4

def test_pca_svd_training_and_projection():
    """Verifies training eigenvectors using first principles SVD."""
    extractor = FeatureExtractor()
    
    # Simulate 3 raw combined vectors of dimension 18148
    raw_feats = np.random.normal(0.0, 1.0, (3, 18148))
    
    # Train PCA keeping k=2 components
    eigenvectors, mean = extractor.train_pca(raw_feats, k=2)
    
    # Assert dimensions
    assert eigenvectors.shape == (2, 18148)
    assert mean.shape == (18148,)
    
    # Test projection of a single vector
    query_vector = raw_feats[0]
    projection = extractor.project_pca(query_vector, eigenvectors, mean)
    assert projection.shape == (2,)
