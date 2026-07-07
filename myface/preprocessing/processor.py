import cv2
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

logger = logging.getLogger("myface.preprocessing")

class PreprocessingConfig:
    """
    Configuration parameters for individual steps in the image preprocessing pipeline.
    """
    def __init__(
        self,
        enable_grayscale: bool = False,
        enable_equalize_hist: bool = False,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        enable_gamma: bool = False,
        gamma: float = 1.0,
        enable_gaussian: bool = False,
        gaussian_ksize: Tuple[int, int] = (5, 5),
        gaussian_sigma: float = 0.0,
        enable_bilateral: bool = True,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        enable_median: bool = False,
        median_ksize: int = 5,
        enable_brightness_norm: bool = False,
        target_brightness: float = 127.0,
        enable_contrast_norm: bool = False,
        target_contrast_std: float = 45.0,
        enable_resize: bool = False,
        resize_dim: Optional[Tuple[int, int]] = None,
        enable_sharpen: bool = False,
        sharpen_strength: float = 1.0
    ):
        self.enable_grayscale = enable_grayscale
        self.enable_equalize_hist = enable_equalize_hist
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size
        self.enable_gamma = enable_gamma
        self.gamma = gamma
        self.enable_gaussian = enable_gaussian
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.enable_bilateral = enable_bilateral
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.enable_median = enable_median
        self.median_ksize = median_ksize
        self.enable_brightness_norm = enable_brightness_norm
        self.target_brightness = target_brightness
        self.enable_contrast_norm = enable_contrast_norm
        self.target_contrast_std = target_contrast_std
        self.enable_resize = enable_resize
        self.resize_dim = resize_dim
        self.enable_sharpen = enable_sharpen
        self.sharpen_strength = sharpen_strength


class ImagePreprocessor:
    """
    Independently configurable preprocessing pipeline executing denoising, 
    illumination normalization, resizing, and sharpening operations.
    """
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, 
            tileGridSize=self.config.clahe_grid_size
        )

    def resize(self, image: np.ndarray, dim: Tuple[int, int]) -> np.ndarray:
        """Resizes the image to target dimensions using area or cubic interpolation."""
        try:
            h, w = image.shape[:2]
            # Use INTER_AREA for downsampling, INTER_CUBIC for upsampling
            interpolation = cv2.INTER_AREA if (dim[0] < w) else cv2.INTER_CUBIC
            return cv2.resize(image, dim, interpolation=interpolation)
        except Exception as e:
            logger.error(f"Image resize operation failed: {e}")
            raise e

    def filter_bilateral(self, image: np.ndarray, d: int, sigma_color: float, sigma_space: float) -> np.ndarray:
        """Applies edge-preserving bilateral filtering to suppress noise."""
        try:
            return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        except Exception as e:
            logger.error(f"Bilateral filter failed: {e}")
            raise e

    def filter_gaussian(self, image: np.ndarray, ksize: Tuple[int, int], sigma: float) -> np.ndarray:
        """Applies Gaussian smoothing kernel to reduce image noise."""
        try:
            return cv2.GaussianBlur(image, ksize, sigma)
        except Exception as e:
            logger.error(f"Gaussian filter failed: {e}")
            raise e

    def filter_median(self, image: np.ndarray, ksize: int) -> np.ndarray:
        """Applies median filtering to eliminate salt-and-pepper noise."""
        try:
            return cv2.medianBlur(image, ksize)
        except Exception as e:
            logger.error(f"Median filter failed: {e}")
            raise e

    def normalize_brightness(self, image: np.ndarray, target: float) -> np.ndarray:
        """Adjusts the image average pixel value to meet a target brightness."""
        try:
            if len(image.shape) == 2:
                current_mean = np.mean(image)
                adjusted = image.astype(np.float32) + (target - current_mean)
                return np.clip(adjusted, 0, 255).astype(np.uint8)
            
            # Apply on Y channel for BGR formats to preserve hue
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            current_mean = np.mean(y)
            y_adj = y.astype(np.float32) + (target - current_mean)
            y_adj = np.clip(y_adj, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([y_adj, cr, cb]), cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.error(f"Brightness normalization failed: {e}")
            raise e

    def normalize_contrast(self, image: np.ndarray, target_std: float) -> np.ndarray:
        """Adjusts the standard deviation of pixel values to match target contrast."""
        try:
            if len(image.shape) == 2:
                mean = np.mean(image)
                std = np.std(image)
                if std == 0:
                    return image
                adjusted = (image.astype(np.float32) - mean) * (target_std / std) + mean
                return np.clip(adjusted, 0, 255).astype(np.uint8)
                
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            mean = np.mean(y)
            std = np.std(y)
            if std == 0:
                return image
            y_adj = (y.astype(np.float32) - mean) * (target_std / std) + mean
            y_adj = np.clip(y_adj, 0, 255).astype(np.uint8)
            return cv2.cvtColor(cv2.merge([y_adj, cr, cb]), cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.error(f"Contrast normalization failed: {e}")
            raise e

    def apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Performs non-linear gamma lookup table mapping."""
        try:
            if gamma == 1.0:
                return image
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            return cv2.LUT(image, table)
        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            raise e

    def equalize_histogram(self, image: np.ndarray) -> np.ndarray:
        """Applies global histogram equalization to spread brightness intensities."""
        try:
            if len(image.shape) == 2:
                return cv2.equalizeHist(image)
                
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = cv2.equalizeHist(y)
            return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.error(f"Histogram equalization failed: {e}")
            raise e

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
        try:
            if len(image.shape) == 2:
                return self.clahe.apply(image)
                
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            y_eq = self.clahe.apply(y)
            return cv2.cvtColor(cv2.merge([y_eq, cr, cb]), cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.error(f"CLAHE execution failed: {e}")
            raise e

    def convert_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Converts BGR image matrix to single-channel Grayscale format."""
        try:
            if len(image.shape) == 2:
                return image
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Grayscale conversion failed: {e}")
            raise e

    def sharpen(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Applies a sharpening convolution filter."""
        try:
            # Custom adjustable Laplacian-based sharpening kernel
            kernel = np.array([
                [0, -1, 0],
                [-1, 4 + strength, -1],
                [0, -1, 0]
            ], dtype=np.float32)
            return cv2.filter2D(image, -1, kernel)
        except Exception as e:
            logger.error(f"Sharpening filter failed: {e}")
            raise e

    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Runs the full raw image matrix through the configured operations.
        
        Args:
            image: Input raw image matrix (BGR).
            
        Returns:
            Dict[str, np.ndarray]: Dict containing the preprocessed 'bgr' (or gray) 
                                   and additional color representations.
        """
        processed = image.copy()
        cfg = self.config

        # 1. Resize
        if cfg.enable_resize and cfg.resize_dim is not None:
            processed = self.resize(processed, cfg.resize_dim)

        # 2. Noise suppression filters
        if cfg.enable_bilateral:
            processed = self.filter_bilateral(
                processed, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
            )
        if cfg.enable_gaussian:
            processed = self.filter_gaussian(processed, cfg.gaussian_ksize, cfg.gaussian_sigma)
        if cfg.enable_median:
            processed = self.filter_median(processed, cfg.median_ksize)

        # 3. Brightness/Contrast Normalization
        if cfg.enable_brightness_norm:
            processed = self.normalize_brightness(processed, cfg.target_brightness)
        if cfg.enable_contrast_norm:
            processed = self.normalize_contrast(processed, cfg.target_contrast_std)

        # 4. Gamma
        if cfg.enable_gamma:
            processed = self.apply_gamma(processed, cfg.gamma)

        # 5. Equalization
        if cfg.enable_clahe:
            processed = self.apply_clahe(processed)
        elif cfg.enable_equalize_hist:
            processed = self.equalize_histogram(processed)

        # 6. Grayscale Conversion
        if cfg.enable_grayscale:
            processed = self.convert_grayscale(processed)

        # 7. Sharpening
        if cfg.enable_sharpen:
            processed = self.sharpen(processed, cfg.sharpen_strength)

        # Build consistent representations dictionary mapping pipeline expectations
        if len(processed.shape) == 2:
            gray = processed
            # Reconstruct dummy BGR representational channel mapping
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            bgr = processed
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

        return {
            "bgr": bgr,
            "gray": gray,
            "ycrcb": ycrcb
        }
