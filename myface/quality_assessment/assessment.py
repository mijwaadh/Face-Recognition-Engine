import cv2
import logging
from typing import Dict, Any, Tuple, Optional, NamedTuple
import numpy as np

logger = logging.getLogger("myface.quality")

class QualityReport(NamedTuple):
    """
    Biometric quality diagnostics metrics summary.
    """
    overall_score: float
    blur_score: float
    focus_score: float
    brightness_score: float
    contrast_score: float
    noise_score: float
    exposure_score: float
    face_size_score: Optional[float]
    pose_score: Optional[float]
    occlusion_score: Optional[float]
    passed: bool
    rejection_reasons: list

class FrameQualityEvaluator:
    """
    Evaluates image and facial properties of frames before entering authentication.
    
    Operates a two-pass validation checks structure:
    - Pass 1 (Image-level): Evaluates focus (Tenengrad), blur (Laplacian), exposure saturation,
      contrast spreads, and noise (Immerkaer method).
    - Pass 2 (Face-level, executed if bbox is provided): Evaluates face size suitability, 
      head pose alignment (symmetry centroid), and occlusions (spatial block deviations).
    """
    def __init__(
        self,
        min_blur_threshold: float = 80.0,
        min_focus_threshold: float = 1000.0,
        min_contrast_threshold: float = 30.0,
        min_exposure_threshold: float = 0.85,
        max_noise_threshold: float = 10.0,
        min_overall_threshold: float = 0.65,
        target_brightness: float = 127.0,
        min_face_size_ratio: float = 0.15,
        max_face_size_ratio: float = 0.60
    ):
        self.min_blur_threshold = min_blur_threshold
        self.min_focus_threshold = min_focus_threshold
        self.min_contrast_threshold = min_contrast_threshold
        self.min_exposure_threshold = min_exposure_threshold
        self.max_noise_threshold = max_noise_threshold
        self.min_overall_threshold = min_overall_threshold
        self.target_brightness = target_brightness
        self.min_face_size_ratio = min_face_size_ratio
        self.max_face_size_ratio = max_face_size_ratio

    def compute_blur_score(self, gray: np.ndarray) -> Tuple[float, float]:
        """Computes Laplacian variance. Higher value means sharper (less blurry)."""
        val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        score = float(1.0 - np.exp(-val / 150.0))
        return score, val

    def compute_focus_score(self, gray: np.ndarray) -> Tuple[float, float]:
        """Computes Tenengrad focus metric (sum of squared Sobel gradients)."""
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        val = float(np.mean(gx**2 + gy**2))
        score = float(1.0 - np.exp(-val / 4000.0))
        return score, val

    def compute_brightness_score(self, gray: np.ndarray) -> Tuple[float, float]:
        """Measures how close average brightness is to target middle gray."""
        val = float(np.mean(gray))
        score = float(1.0 - (abs(val - self.target_brightness) / 127.0))
        score = max(0.0, min(1.0, score))
        return score, val

    def compute_contrast_score(self, gray: np.ndarray) -> Tuple[float, float]:
        """Measures intensity standard deviation relative to target contrast spread."""
        val = float(np.std(gray))
        score = float(min(1.0, val / 50.0))
        return score, val

    def estimate_noise(self, gray: np.ndarray) -> Tuple[float, float]:
        """Estimates pixel noise standard deviation using Immerkaer's fast algorithm."""
        # Laplacian-like noise estimation mask
        kernel = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ], dtype=np.float32)
        
        noise_map = cv2.filter2D(gray, cv2.CV_32F, kernel)
        h, w = gray.shape
        # Average absolute difference normalization constant
        sum_abs = float(np.sum(np.abs(noise_map)))
        noise_std = sum_abs * np.sqrt(0.5 * np.pi) / (6.0 * (w - 2) * (h - 2))
        
        score = float(max(0.0, 1.0 - (noise_std / 15.0)))
        return score, noise_std

    def compute_exposure_quality(self, gray: np.ndarray) -> Tuple[float, float]:
        """Measures saturation ratio of pixels. Returns ratio of correctly exposed pixels."""
        under = np.sum(gray < 15)
        over = np.sum(gray > 240)
        saturated_ratio = float((under + over) / gray.size)
        score = 1.0 - saturated_ratio
        return score, saturated_ratio

    def evaluate_face_size(self, bbox: Any, image_shape: Tuple[int, int]) -> float:
        """Measures face crop area suitability relative to full image area."""
        img_h, img_w = image_shape[:2]
        face_area = bbox.w * bbox.h
        img_area = img_w * img_h
        ratio = face_area / img_area
        
        if self.min_face_size_ratio <= ratio <= self.max_face_size_ratio:
            return 1.0
        # Linear penalty if size falls outside bounds
        if ratio < self.min_face_size_ratio:
            return float(ratio / self.min_face_size_ratio)
        return float((1.0 - ratio) / (1.0 - self.max_face_size_ratio))

    def evaluate_pose_quality(self, gray_face: np.ndarray) -> float:
        """Evaluates frontal pose quality using intensity centroid offset."""
        try:
            h, w = gray_face.shape
            M = cv2.moments(gray_face)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                offset_x = (cx - (w / 2.0)) / (w / 2.0)
                # Frontal pose score drops as offset increases
                score = float(max(0.0, 1.0 - abs(offset_x) * 3.0))
                return score
            return 0.5
        except Exception:
            return 0.5

    def evaluate_occlusion(self, gray_face: np.ndarray) -> float:
        """
        Estimates face occlusion presence by dividing the face into 16 blocks 
        and flagging low-variance (flat) textureless zones.
        """
        h, w = gray_face.shape
        bh, bw = h // 4, w // 4
        flat_blocks = 0
        
        for r in range(4):
            for c in range(4):
                block = gray_face[r*bh:(r+1)*bh, c*bw:(c+1)*bw]
                if block.size > 0:
                    std = np.std(block)
                    # Flat blocks (low variance, e.g. hand, mask, or flat occlusion block)
                    if std < 8.0:
                        flat_blocks += 1
                        
        occlusion_ratio = flat_blocks / 16.0
        score = float(1.0 - occlusion_ratio)
        return score

    def evaluate_frame(self, image: np.ndarray, bbox: Optional[Any] = None) -> QualityReport:
        """
        Runs image-level diagnostics and (if bbox provided) facial diagnostics checks.
        
        Args:
            image: BGR input image frame.
            bbox: Optional face bounding box (namedtuple or class exposing x, y, w, h).
            
        Returns:
            QualityReport: Diagnosed metrics, fused score, and validation status.
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            rejection_reasons = []

            # Pass 1: Image level
            b_score, b_val = self.compute_blur_score(gray)
            f_score, f_val = self.compute_focus_score(gray)
            bright_score, bright_val = self.compute_brightness_score(gray)
            contrast_score, contrast_val = self.compute_contrast_score(gray)
            noise_score, noise_val = self.estimate_noise(gray)
            exp_score, exp_val = self.compute_exposure_quality(gray)

            # Assert thresholds
            if b_val < self.min_blur_threshold:
                rejection_reasons.append(f"Image too blurry (Laplacian var {b_val:.1f} < {self.min_blur_threshold})")
            if f_val < self.min_focus_threshold:
                rejection_reasons.append(f"Out of focus (Tenengrad {f_val:.1f} < {self.min_focus_threshold})")
            if contrast_val < self.min_contrast_threshold:
                rejection_reasons.append(f"Low contrast (Std {contrast_val:.1f} < {self.min_contrast_threshold})")
            if exp_score < self.min_exposure_threshold:
                rejection_reasons.append(f"Bad exposure (Exposure score {exp_score:.2f} < {self.min_exposure_threshold})")
            if noise_val > self.max_noise_threshold:
                rejection_reasons.append(f"High noise (Noise std {noise_val:.1f} > {self.max_noise_threshold})")

            # Base score fusion
            base_score = float((b_score + f_score + bright_score + contrast_score + noise_score + exp_score) / 6.0)

            # Pass 2: Face level (if bbox provided)
            face_size_score = None
            pose_score = None
            occlusion_score = None
            
            if bbox is not None:
                face_size_score = self.evaluate_face_size(bbox, image.shape)
                
                # Extract crop
                y_start = max(0, bbox.y)
                y_end = min(image.shape[0], bbox.y + bbox.h)
                x_start = max(0, bbox.x)
                x_end = min(image.shape[1], bbox.x + bbox.w)
                
                crop = gray[y_start:y_end, x_start:x_end]
                
                if crop.size > 0:
                    pose_score = self.evaluate_pose_quality(crop)
                    occlusion_score = self.evaluate_occlusion(crop)
                else:
                    pose_score = 0.0
                    occlusion_score = 0.0

                if face_size_score < 0.5:
                    rejection_reasons.append(f"Unsuitable face size ratio (Score {face_size_score:.2f} < 0.5)")
                if pose_score < 0.5:
                    rejection_reasons.append(f"Excessive profile pose yaw (Pose score {pose_score:.2f} < 0.5)")
                if occlusion_score < 0.7:
                    rejection_reasons.append(f"Face occlusion detected (Occlusion score {occlusion_score:.2f} < 0.7)")

                # Weighted final fusion combining image and face assessments
                overall_score = float(0.4 * base_score + 0.2 * face_size_score + 0.2 * pose_score + 0.2 * occlusion_score)
            else:
                overall_score = base_score

            if overall_score < self.min_overall_threshold:
                rejection_reasons.append(f"Overall quality score {overall_score:.2f} below target {self.min_overall_threshold}")

            passed = len(rejection_reasons) == 0

            return QualityReport(
                overall_score=round(overall_score, 4),
                blur_score=round(b_score, 4),
                focus_score=round(f_score, 4),
                brightness_score=round(bright_score, 4),
                contrast_score=round(contrast_score, 4),
                noise_score=round(noise_score, 4),
                exposure_score=round(exp_score, 4),
                face_size_score=round(face_size_score, 4) if face_size_score is not None else None,
                pose_score=round(pose_score, 4) if pose_score is not None else None,
                occlusion_score=round(occlusion_score, 4) if occlusion_score is not None else None,
                passed=passed,
                rejection_reasons=rejection_reasons
            )
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            raise e
