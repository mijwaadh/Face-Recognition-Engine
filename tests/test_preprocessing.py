import pytest
import numpy as np
import cv2
from myface.preprocessing.processor import ImagePreprocessor, PreprocessingConfig

def test_grayscale_conversion():
    """Verifies that conversion to grayscale correctly maps channels."""
    config = PreprocessingConfig(enable_grayscale=True, enable_clahe=False, enable_bilateral=False)
    preprocessor = ImagePreprocessor(config=config)
    
    # 3-channel input
    img = np.ones((50, 50, 3), dtype=np.uint8) * 100
    res = preprocessor.preprocess(img)
    
    assert len(res["gray"].shape) == 2
    assert res["gray"].shape == (50, 50)
    assert np.all(res["gray"] == 100)

def test_image_resizing():
    """Verifies that resizing changes dimensions as configured."""
    config = PreprocessingConfig(enable_resize=True, resize_dim=(30, 20), enable_clahe=False, enable_bilateral=False)
    preprocessor = ImagePreprocessor(config=config)
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    res = preprocessor.preprocess(img)
    
    assert res["bgr"].shape[:2] == (20, 30)
    assert res["gray"].shape == (20, 30)

def test_brightness_and_contrast_normalization():
    """Verifies shift/scale targets for average brightness and contrast std."""
    config = PreprocessingConfig(
        enable_brightness_norm=True,
        target_brightness=150.0,
        enable_contrast_norm=True,
        target_contrast_std=30.0,
        enable_clahe=False,
        enable_bilateral=False
    )
    preprocessor = ImagePreprocessor(config=config)
    
    # Use a random normal canvas centered around 100 with std 10 to allow scaling without clipping
    np.random.seed(42)
    noise = np.random.normal(100.0, 10.0, (50, 50, 3))
    img = np.clip(noise, 0, 255).astype(np.uint8)
    
    res = preprocessor.preprocess(img)
    mean_val = np.mean(res["gray"])
    std_val = np.std(res["gray"])
    
    # Brightness should be close to 150.0
    assert abs(mean_val - 150.0) < 2.0
    # Contrast should be scaled close to 30.0
    assert abs(std_val - 30.0) < 2.0

def test_gamma_correction():
    """Verifies look-up table non-linear intensity mapping."""
    config = PreprocessingConfig(enable_gamma=True, gamma=2.0, enable_clahe=False, enable_bilateral=False)
    preprocessor = ImagePreprocessor(config=config)
    
    img = np.ones((10, 10, 3), dtype=np.uint8) * 100
    res = preprocessor.preprocess(img)
    
    # Gamma 2.0 (gamma > 1) shifts values upwards
    assert res["gray"][0, 0] > 100

def test_sharpening_filter():
    """Verifies that high-pass sharpening increases localized edge gradients."""
    config = PreprocessingConfig(enable_sharpen=True, sharpen_strength=2.0, enable_clahe=False, enable_bilateral=False)
    preprocessor = ImagePreprocessor(config=config)
    
    # Draw a step edge
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, 5:] = 100
    
    res = preprocessor.preprocess(img)
    # Sharpening must overshoot local gradients at edges
    assert res["gray"][5, 4] < 0 or res["gray"][5, 5] > 100

def test_all_filters():
    """Verifies sequential filtering runs successfully without compile/runtime errors."""
    config = PreprocessingConfig(
        enable_gaussian=True,
        enable_bilateral=True,
        enable_median=True,
        enable_clahe=True,
        enable_equalize_hist=False
    )
    preprocessor = ImagePreprocessor(config=config)
    
    img = np.ones((50, 50, 3), dtype=np.uint8) * 128
    res = preprocessor.preprocess(img)
    
    assert res["bgr"].shape == (50, 50, 3)
    assert res["gray"].shape == (50, 50)
