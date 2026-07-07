import cv2
import logging
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger("myface.anti_spoof")

class LivenessDetector:
    """
    Computes passive biometric anti-spoofing scores using classical image analysis.
    
    Combines:
    1. Shape-from-Shading (SfS): Solves the Poisson equation via 2D FFT to reconstruct 
       relative depth Z and computes its variance (3D relief).
    2. FFT Moiré Pattern Detection: Identifies high-frequency impulses in the 2D FFT 
       log-magnitude spectrum of screens.
    3. Chrominance Diffusion: Measures the standard deviation of Cr/Cb color gradients 
       to identify print halftone patterns.
    4. LBP Texture: Evaluates spatial micro-textures using Local Binary Patterns variance.
    5. Laplacian Variance: Evaluates localized sharpness.
    6. Farneback Optical Flow: Measures non-rigid motion variance across frames to detect 
       2D static spoofs vs 3D living motion.
    7. Logistic Regression Fusion: Integrates individual scores using a Sigmoid probability:
         P(live) = 1 / (1 + exp(-(W^T * X + bias)))
    """
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        bias: float = -4.5
    ):
        # Default pre-calibrated weights for [SfS, Moiré (inverse), Diffusion, LBP, Sharpness, Flow]
        self.weights = weights or [2.5, 3.0, 1.5, 1.0, 1.0, 1.0]
        self.bias = bias

    def compute_sfs_depth_variance(self, gray_face: np.ndarray) -> float:
        """
        Reconstructs face relative depth using a fast Poisson solver in FFT frequency domain.
        
        Formula:
          Z_FFT(u, v) = -F_FFT(u, v) / (4 * pi^2 * (u^2 + v^2))
        where F is the divergence of horizontal and vertical gradients.
        """
        try:
            # 1. Compute gradients
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            
            # 2. Compute divergence: f = d(grad_x)/dx + d(grad_y)/dy
            f = cv2.Sobel(grad_x, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(grad_y, cv2.CV_64F, 0, 1, ksize=3)
            
            h, w = gray_face.shape
            F_fft = np.fft.fft2(f)
            
            # 3. Frequency grids
            u, v = np.meshgrid(np.fft.fftfreq(w), np.fft.fftfreq(h))
            denom = 4.0 * np.pi**2 * (u**2 + v**2)
            denom[0, 0] = 1.0  # Avoid zero-division at DC component
            
            # 4. Integrate gradients
            Z_fft = -F_fft / denom
            Z_fft[0, 0] = 0.0  # Set DC component to 0
            
            Z = np.real(np.fft.ifft2(Z_fft))
            
            # 5. Measure depth variance (live face has curvature, spoof is flat)
            depth_var = float(np.var(Z))
            
            # Normalize variance to [0, 1] range
            score = float(1.0 - np.exp(-depth_var * 1e3))
            return score
        except Exception as e:
            logger.warning(f"SfS depth calculation failed: {e}")
            return 0.5

    def detect_moire_fft_peaks(self, gray_face: np.ndarray) -> float:
        """
        Scans for Screen Moiré patterns (impulse peaks) in 2D FFT magnitude spectrum.
        """
        try:
            h, w = gray_face.shape
            F = np.fft.fft2(gray_face)
            F_shift = np.fft.fftshift(F)
            mag = np.log(1.0 + np.abs(F_shift))
            
            # Mask central low frequencies (radius 12)
            cy, cx = h // 2, w // 2
            y_indices, x_indices = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            
            outer_mask = dist_from_center > 12
            outer_mag = mag[outer_mask]
            
            if outer_mag.size == 0:
                return 0.0
                
            mean = np.mean(outer_mag)
            std = np.std(outer_mag)
            
            # Flag pixels that are 4.5 standard deviations above the mean (aliasing peaks)
            threshold = mean + 4.5 * std
            peaks_count = np.sum(outer_mag > threshold)
            
            score = float(min(1.0, peaks_count / 15.0))
            return score
        except Exception as e:
            logger.warning(f"FFT Moire detection failed: {e}")
            return 0.0

    def compute_chrominance_diffusion(self, bgr_face: np.ndarray) -> float:
        """
        Measures skin-tone chrominance gradient variance to detect print halftone grids.
        """
        try:
            ycrcb = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)
            
            # Compute gradients in Cr and Cb
            dx_cr = cv2.Sobel(cr, cv2.CV_64F, 1, 0, ksize=3)
            dy_cr = cv2.Sobel(cr, cv2.CV_64F, 0, 1, ksize=3)
            
            mag_cr = np.sqrt(dx_cr**2 + dy_cr**2)
            
            # Smooth skin chrominance variance is low (high score), print is high (low score)
            std_val = np.std(mag_cr)
            score = float(1.0 - min(1.0, std_val / 20.0))
            return score
        except Exception as e:
            logger.warning(f"Chrominance diffusion failed: {e}")
            return 0.5

    def analyze_lbp_texture(self, gray_face: np.ndarray) -> float:
        """
        Evaluates micro-textures using Local Binary Pattern (LBP) histogram variance.
        """
        try:
            h, w = gray_face.shape
            lbp_img = np.zeros((h - 2, w - 2), dtype=np.uint8)
            center = gray_face[1:-1, 1:-1]
            
            neighbors = [
                gray_face[0:-2, 0:-2], gray_face[0:-2, 1:-1], gray_face[0:-2, 2:],
                gray_face[1:-1, 2:], gray_face[2:, 2:], gray_face[2:, 1:-1],
                gray_face[2:, 0:-2], gray_face[1:-1, 0:-2]
            ]
            for p, n in enumerate(neighbors):
                lbp_img += ((n >= center).astype(np.uint8) << p)
                
            hist, _ = np.histogram(lbp_img, bins=range(257))
            hist_norm = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
            
            # Real skin micro-texture histogram displays high variance (peaks at uniform codes)
            variance = np.var(hist_norm)
            score = float(min(1.0, variance * 1000.0))
            return score
        except Exception as e:
            logger.warning(f"LBP texture check failed: {e}")
            return 0.5

    def compute_optical_flow_magnitude(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """
        Computes the ratio of non-rigid motion using Farneback dense optical flow.
        
        Spoof printed photos display uniform rigid translation (low variance relative to mean).
        """
        try:
            # Rescale to match dims if different
            if prev_gray.shape != curr_gray.shape:
                prev_gray = cv2.resize(prev_gray, (curr_gray.shape[1], curr_gray.shape[0]))
                
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            mean_motion = np.mean(mag)
            std_motion = np.std(mag)
            
            if mean_motion < 0.05:
                # Flat/static frame transition (no motion is neutral/low liveness)
                return 0.2
                
            # Non-rigidity ratio: standard deviation / mean
            non_rigidity = std_motion / (mean_motion + 1e-6)
            score = float(min(1.0, non_rigidity / 2.0))
            return score
        except Exception as e:
            logger.warning(f"Optical flow magnitude failed: {e}")
            return 0.5

    def analyze_liveness(
        self,
        bgr_face: np.ndarray,
        threshold: float,
        prev_gray: Optional[np.ndarray] = None
    ) -> Tuple[bool, float]:
        """
        Fuses anti-spoofing scores using a pre-calibrated Logistic Sigmoid model.
        
        Args:
            bgr_face: Aligned color face matrix (128x128).
            threshold: Liveness classification threshold.
            prev_gray: Optional previous frame in grayscale for optical flow.
            
        Returns:
            Tuple[bool, float]: (is_live, liveness_probability)
        """
        try:
            gray = cv2.cvtColor(bgr_face, cv2.COLOR_BGR2GRAY)
            
            # 1. Shape-from-Shading relative depth variance
            sfs_val = self.compute_sfs_depth_variance(gray)
            
            # 2. FFT screen moire pattern peak scan
            moire_val = self.detect_moire_fft_peaks(gray)
            
            # 3. Chrominance diffusion gradient variance
            diff_val = self.compute_chrominance_diffusion(bgr_face)
            
            # 4. LBP micro-texture histogram variance
            lbp_val = self.analyze_lbp_texture(gray)
            
            # 5. Laplacian variance sharpness
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sharpness_val = float(min(1.0, lap_var / 1200.0))
            
            # 6. Optical Flow motion non-rigidity
            if prev_gray is not None:
                flow_val = self.compute_optical_flow_magnitude(prev_gray, gray)
            else:
                flow_val = 0.5  # Neutral default score for single frame requests
                
            # Compile feature score vector x
            # Note: Moiré is inverted since high moire peaks represents screen spoofing
            x = np.array([
                sfs_val,
                1.0 - moire_val,
                diff_val,
                lbp_val,
                sharpness_val,
                flow_val
            ], dtype=np.float32)
            
            # 7. Logistic Sigmoid Fusion: P = 1 / (1 + exp(-(W^T * X + bias)))
            logit = float(np.dot(self.weights, x) + self.bias)
            liveness_prob = float(1.0 / (1.0 + np.exp(-logit)))
            
            is_live = liveness_prob >= threshold
            
            logger.info(
                f"Liveness Check: {is_live} | P(live)={liveness_prob:.4f} | "
                f"Features: SfS={sfs_val:.2f}, Moire={moire_val:.2f}, Diff={diff_val:.2f}, "
                f"LBP={lbp_val:.2f}, Sharp={sharpness_val:.2f}, Flow={flow_val:.2f}"
            )
            return is_live, liveness_prob
        except Exception as e:
            logger.error(f"Liveness analysis failed: {e}")
            raise e
