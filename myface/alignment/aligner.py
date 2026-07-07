import cv2
import logging
from typing import Tuple, Optional, Dict
import numpy as np
from myface.detection.detector import BBox

logger = logging.getLogger("myface.alignment")

class FaceAligner:
    """
    Normalizes face geometry using eye center locations and 2D affine warping.
    
    Uses personalized left and right eye templates enrolled during training 
    to identify eye centers, rotate the face so the eyes are horizontal,
    and crop/resize to target dimensions.
    
    Mathematical Principles:
    1. Eye Center Tracking:
       The pupil/iris is darker than the surrounding skin. We identify dark valleys 
       (minimum pixel values) in horizontal left and right regions.
    2. NCC Search:
       For query images, left/right search windows are extracted around the estimated coordinates,
       and Normalized Cross-Correlation (cv2.matchTemplate with TM_CCOEFF_NORMED) locates
       the template matches maximizing correlation.
    3. Similarity Transform Map:
       For eye midpoint C, distance D, angle theta = arctan2(dy, dx), and scale = D_target / D,
       we compute the 2D affine warp matrix mapping the eyes horizontally to standard 
       geometric coordinates: Left Eye (target_width * 0.325, target_height * 0.35) and 
       Right Eye (target_width * 0.675, target_height * 0.35).
    """
    def __init__(self, target_size: int = 128, eye_distance_ratio: float = 0.35):
        self.target_size = target_size
        self.eye_distance_ratio = eye_distance_ratio
        self.template_radius = 12

    def extract_eye_templates(self, gray_image: np.ndarray, bbox: BBox) -> Dict[str, np.ndarray]:
        """
        Locates pupil centers using intensity valleys and extracts 24x24 patches.
        
        Args:
            gray_image: Grayscale source frame matrix.
            bbox: BBox coordinates of the face.
            
        Returns:
            Dict[str, np.ndarray]: Dict containing 'left_eye' and 'right_eye' patches.
        """
        try:
            # 1. Define left and right eye search sub-zones inside the top 45% of the bbox
            y_start = int(bbox.y + bbox.h * 0.20)
            y_end = int(bbox.y + bbox.h * 0.45)
            
            left_x_start = int(bbox.x + bbox.w * 0.12)
            left_x_end = int(bbox.x + bbox.w * 0.48)
            
            right_x_start = int(bbox.x + bbox.w * 0.52)
            right_x_end = int(bbox.x + bbox.w * 0.88)
            
            # Boundary checks
            y_start = max(0, min(y_start, gray_image.shape[0] - 1))
            y_end = max(y_start + 5, min(y_end, gray_image.shape[0]))
            
            left_x_start = max(0, min(left_x_start, gray_image.shape[1] - 1))
            left_x_end = max(left_x_start + 5, min(left_x_end, gray_image.shape[1]))
            
            right_x_start = max(0, min(right_x_start, gray_image.shape[1] - 1))
            right_x_end = max(right_x_start + 5, min(right_x_end, gray_image.shape[1]))
            
            left_crop = gray_image[y_start:y_end, left_x_start:left_x_end]
            right_crop = gray_image[y_start:y_end, right_x_start:right_x_end]
            
            # 2. Identify dark intensity valleys (pupil center candidates)
            _, _, min_loc_l, _ = cv2.minMaxLoc(left_crop)
            _, _, min_loc_r, _ = cv2.minMaxLoc(right_crop)
            
            left_eye_center = (left_x_start + min_loc_l[0], y_start + min_loc_l[1])
            right_eye_center = (right_x_start + min_loc_r[0], y_start + min_loc_r[1])
            
            # 3. Crop 24x24 patches around pupils
            r = self.template_radius
            
            left_eye_patch = self._safe_crop(gray_image, left_eye_center, r)
            right_eye_patch = self._safe_crop(gray_image, right_eye_center, r)
            
            return {
                "left_eye": left_eye_patch,
                "right_eye": right_eye_patch
            }
        except Exception as e:
            logger.error(f"Failed to extract eye templates: {e}")
            # Fallback to standard black arrays
            return {
                "left_eye": np.zeros((r*2, r*2), dtype=np.uint8),
                "right_eye": np.zeros((r*2, r*2), dtype=np.uint8)
            }

    def _safe_crop(self, image: np.ndarray, center: Tuple[int, int], radius: int) -> np.ndarray:
        """Helper to crop patch safely, handling border boundaries."""
        cx, cy = center
        y1 = max(0, cy - radius)
        y2 = min(image.shape[0], cy + radius)
        x1 = max(0, cx - radius)
        x2 = min(image.shape[1], cx + radius)
        
        crop = image[y1:y2, x1:x2]
        target_dim = radius * 2
        if crop.shape[0] != target_dim or crop.shape[1] != target_dim:
            # Resize or pad if crop got clipped near image borders
            return cv2.resize(crop, (target_dim, target_dim))
        return crop

    def locate_eyes(
        self,
        gray_image: np.ndarray,
        bbox: BBox,
        templates: Dict[str, np.ndarray]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Locates eye centers in a query frame using template matching correlation.
        
        Args:
            gray_image: Grayscale search image.
            bbox: Face localization box.
            templates: Dict containing 'left_eye' and 'right_eye' templates.
            
        Returns:
            Tuple[Tuple[int,int], Tuple[int,int]]: Left and right eye center coordinates.
        """
        r = self.template_radius
        
        # Define search zones (slightly wider than template extraction to handle tilts)
        y_start = int(bbox.y + bbox.h * 0.05)
        y_end = int(bbox.y + bbox.h * 0.70)
        
        left_x_start = int(bbox.x + bbox.w * 0.02)
        left_x_end = int(bbox.x + bbox.w * 0.55)
        
        right_x_start = int(bbox.x + bbox.w * 0.45)
        right_x_end = int(bbox.x + bbox.w * 0.98)
        
        # Boundary limits
        y_start = max(0, min(y_start, gray_image.shape[0] - 1))
        y_end = max(y_start + 10, min(y_end, gray_image.shape[0]))
        left_x_start = max(0, min(left_x_start, gray_image.shape[1] - 1))
        left_x_end = max(left_x_start + 10, min(left_x_end, gray_image.shape[1]))
        right_x_start = max(0, min(right_x_start, gray_image.shape[1] - 1))
        right_x_end = max(right_x_start + 10, min(right_x_end, gray_image.shape[1]))
        
        left_zone = gray_image[y_start:y_end, left_x_start:left_x_end]
        right_zone = gray_image[y_start:y_end, right_x_start:right_x_end]
        
        # Normalized cross correlation search
        left_eye_center = (int(bbox.x + bbox.w * 0.35), int(bbox.y + bbox.h * 0.35))
        right_eye_center = (int(bbox.x + bbox.w * 0.65), int(bbox.y + bbox.h * 0.35))
        
        try:
            if left_zone.shape[0] >= templates["left_eye"].shape[0] and left_zone.shape[1] >= templates["left_eye"].shape[1]:
                res_l = cv2.matchTemplate(left_zone, templates["left_eye"], cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc_l = cv2.minMaxLoc(res_l)
                left_eye_center = (left_x_start + max_loc_l[0] + r, y_start + max_loc_l[1] + r)
                
            if right_zone.shape[0] >= templates["right_eye"].shape[0] and right_zone.shape[1] >= templates["right_eye"].shape[1]:
                res_r = cv2.matchTemplate(right_zone, templates["right_eye"], cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc_r = cv2.minMaxLoc(res_r)
                right_eye_center = (right_x_start + max_loc_r[0] + r, y_start + max_loc_r[1] + r)
        except Exception as e:
            logger.warning(f"Template match correlation search failed: {e}. Falling back to geometry ratios.")
            
        return left_eye_center, right_eye_center

    def align(self, image: np.ndarray, bbox: BBox, templates: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """
        Aligns, rotates, and crops the face image using eye coordinates similarity warping.
        
        Args:
            image: Preprocessed grayscale or BGR matrix.
            bbox: Face localization box coordinates.
            templates: Optional dictionary of enrolled eye templates.
            
        Returns:
            np.ndarray: Aligned grayscale crop of shape (target_size, target_size).
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # 1. Locate left and right eye centers
            if templates is not None and "left_eye" in templates and "right_eye" in templates:
                left_eye, right_eye = self.locate_eyes(gray, bbox, templates)
            else:
                # Fallback to standard proportions if templates are missing
                left_eye = (int(bbox.x + bbox.w * 0.325), int(bbox.y + bbox.h * 0.35))
                right_eye = (int(bbox.x + bbox.w * 0.675), int(bbox.y + bbox.h * 0.35))

            # 2. Compute similarity parameters: midpoint, angle, distance scale
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            
            angle = float(np.arctan2(dy, dx) * 180.0 / np.pi)
            center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))
            
            dist = np.sqrt(dx**2 + dy**2)
            target_dist = self.target_size * self.eye_distance_ratio
            scale = float(target_dist / dist) if dist > 0 else 1.0

            # 3. Formulate similarity transform matrix
            M = cv2.getRotationMatrix2D(center, angle, scale)

            # 4. Map midpoint to standard coordinates: (target_width/2, target_height*0.35)
            target_cx = self.target_size / 2.0
            target_cy = self.target_size * 0.35
            
            M[0, 2] += target_cx - center[0]
            M[1, 2] += target_cy - center[1]

            # 5. Warp, normalize scale, and crop
            aligned = cv2.warpAffine(
                gray, 
                M, 
                (self.target_size, self.target_size), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.info("Aligned face using affine transformation matrix.")
            return aligned
        except Exception as e:
            logger.error(f"Alignment operation failed: {e}")
            raise e
