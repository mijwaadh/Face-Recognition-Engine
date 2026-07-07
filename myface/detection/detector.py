import cv2
import logging
from typing import Optional, NamedTuple, List, Tuple
import numpy as np

logger = logging.getLogger("myface.detection")

class BBox(NamedTuple):
    """
    Representation of a localized face region bounding box.
    """
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    """
    Face detector utilizing Bilateral Symmetry Mapping, custom HOG Template Matching, 
    and Vertical Projection Profile Verification.
    
    Mathematical Principles:
    1. Bilateral Symmetry Map:
       A human face is highly symmetrical. We compute the horizontal gradient GradX.
       Across the vertical symmetry axis c, left and right gradients have opposite signs:
       GradX(c - w, y) = -GradX(c + w, y).
       We calculate the Normalized Gradient Symmetry Score at column c and scale W:
         S(c) = Sum_y Sum_w ( -GradX(c - w, y) * GradX(c + w, y) ) / (Norm(L) * Norm(R))
    2. HOG Template Correlation:
       A synthetic face representation is programmatically drawn (head oval, eyes/mouth lines).
       Its HOG descriptor is extracted to act as a 'mean face' HOG template.
       Query candidates are cropped, resized to 64x64, their HOG descriptors are extracted,
       and Cosine Similarity against the template is computed.
    3. Projection Profile Verification:
       Horizontal projection of vertical gradients GradY (sum along rows) displays distinct peaks
       at the eye row and mouth row, and a dip at the nose bridge. Candidates that do not
       exhibit this structural valley profile are rejected.
    """
    def __init__(
        self,
        min_size: int = 80,
        max_size: int = 240,
        aspect_ratio: float = 1.4,
        confidence_threshold: float = 0.65,
        nms_iou_threshold: float = 0.3
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratio = aspect_ratio
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        
        # Build the master synthetic face HOG template at initialization
        self.template_dim = 64
        self.face_template_hog = self._build_synthetic_face_template()

    def _build_synthetic_face_template(self) -> np.ndarray:
        """
        Draws a geometric face contour and extracts its HOG descriptor.
        """
        dim = self.template_dim
        canvas = np.zeros((dim, dim), dtype=np.uint8)
        
        # Draw head ellipse: Center (32, 32), axes (22, 28)
        cv2.ellipse(canvas, (32, 32), (22, 28), 0, 0, 360, 200, -1)
        # Draw eyes: horizontal bar at y=24, x from 20 to 44
        cv2.line(canvas, (20, 24), (30, 24), 80, 2)
        cv2.line(canvas, (34, 24), (44, 24), 80, 2)
        # Draw nose: vertical bar at x=32, y from 26 to 42
        cv2.line(canvas, (32, 26), (32, 42), 120, 2)
        # Draw mouth: horizontal bar at y=48, x from 22 to 42
        cv2.line(canvas, (22, 48), (42, 48), 60, 2)
        
        # Smooth canvas to model natural gradient transitions
        canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
        return self._extract_hog(canvas)

    def _extract_hog(self, gray_64x64: np.ndarray) -> np.ndarray:
        """
        Extracts HOG descriptor from a 64x64 grayscale crop from scratch.
        
        Cell size: 8x8 pixels (8x8 cells). Block size: 2x2 cells (overlap 1 cell).
        Orientations: 9 bins (0 to 180 degrees).
        """
        # 1. Compute Sobel gradients
        gx = cv2.Sobel(gray_64x64, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_64x64, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180.0 / np.pi)
        angle = np.where(angle < 0, angle + 180.0, angle) # Unsigned gradients [0, 180]
        
        h, w = gray_64x64.shape
        cell_size = 8
        num_cells_y = h // cell_size
        num_cells_x = w // cell_size
        num_bins = 9
        
        # 2. Compute histograms for each cell
        histograms = np.zeros((num_cells_y, num_cells_x, num_bins), dtype=np.float32)
        bin_width = 180.0 / num_bins
        
        for cy in range(num_cells_y):
            for cx in range(num_cells_x):
                # Crop cell regions
                cell_mag = magnitude[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                cell_ang = angle[cy*cell_size:(cy+1)*cell_size, cx*cell_size:(cx+1)*cell_size]
                
                # Bilinear bin voting based on angle
                bins_float = cell_ang / bin_width
                bin_idx1 = np.floor(bins_float).astype(np.int32) % num_bins
                bin_idx2 = (bin_idx1 + 1) % num_bins
                
                weight2 = bins_float - np.floor(bins_float)
                weight1 = 1.0 - weight2
                
                # Accumulate weighted magnitude
                np.add.at(histograms[cy, cx], bin_idx1, cell_mag * weight1)
                np.add.at(histograms[cy, cx], bin_idx2, cell_mag * weight2)
                
        # 3. Block normalization (2x2 cells block, overlap 1 cell)
        block_feats = []
        epsilon = 1e-6
        for by in range(num_cells_y - 1):
            for bx in range(num_cells_x - 1):
                # Gather 2x2 cells histograms
                block = histograms[by:by+2, bx:bx+2].flatten()
                # L2-Norm normalization
                norm = np.sqrt(np.sum(block**2) + epsilon**2)
                block_feats.append(block / norm)
                
        return np.concatenate(block_feats)

    def compute_symmetry_map(self, gray: np.ndarray, scale_w: int) -> np.ndarray:
        """
        Computes horizontal gradient reflection symmetry score across columns.
        """
        h, w = gray.shape
        # Calculate horizontal gradients
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        
        symmetry_scores = np.zeros((w,), dtype=np.float32)
        half_scale = scale_w // 2
        
        # Slide symmetry axis column c
        start_col = max(half_scale, 20)
        end_col = min(w - half_scale, w - 20)
        
        for c in range(start_col, end_col):
            # Crop left and right columns relative to axis c
            left = grad_x[:, c - half_scale : c]
            right = grad_x[:, c : c + half_scale]
            
            # Flip right crop horizontally
            right_flipped = -np.fliplr(right) # Opposite gradient signs reflection
            
            # Compute correlation: dot product sum(L * R)
            dot = np.sum(left * right_flipped)
            norm_l = np.sqrt(np.sum(left**2))
            norm_r = np.sqrt(np.sum(right_flipped**2))
            
            if norm_l > 0 and norm_r > 0:
                symmetry_scores[c] = dot / (norm_l * norm_r)
                
        return symmetry_scores

    def verify_projection_profile(self, gray_crop: np.ndarray) -> float:
        """
        Verifies the vertical projection of horizontal edges (GradY).
        
        Real faces display an eye-row peak, a nose-bridge valley, and a mouth-row peak.
        """
        try:
            # Sobel in Y axis (horizontal edges)
            grad_y = cv2.Sobel(gray_crop, cv2.CV_32F, 0, 1, ksize=3)
            # Sum rows: vertical projection profile
            row_sums = np.sum(np.abs(grad_y), axis=1)
            h = len(row_sums)
            
            # Split profile: upper half (eyes), middle (nose bridge), bottom (mouth)
            upper_profile = row_sums[:int(h * 0.45)]
            middle_profile = row_sums[int(h * 0.45):int(h * 0.70)]
            bottom_profile = row_sums[int(h * 0.70):]
            
            if len(upper_profile) == 0 or len(middle_profile) == 0 or len(bottom_profile) == 0:
                return 0.0
                
            peak_upper = np.max(upper_profile)
            valley_middle = np.min(middle_profile)
            peak_bottom = np.max(bottom_profile)
            
            # Structured faces satisfy: Peak Upper > Valley Middle and Peak Bottom > Valley Middle
            if peak_upper > valley_middle and peak_bottom > valley_middle:
                # Calculate ratio score
                score = 1.0 - (valley_middle / (max(peak_upper, peak_bottom) + 1e-5))
                return float(np.clip(score, 0.0, 1.0))
            return 0.0
        except Exception:
            return 0.0

    def _compute_iou(self, box1: BBox, box2: BBox) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        x1 = max(box1.x, box2.x)
        y1 = max(box1.y, box2.y)
        x2 = min(box1.x + box1.w, box2.x + box2.w)
        y2 = min(box1.y + box1.h, box2.y + box2.h)
        
        inter_w = max(0, x2 - x1)
        inter_h = max(0, y2 - y1)
        intersection = inter_w * inter_h
        
        union = (box1.w * box1.h) + (box2.w * box2.h) - intersection
        if union == 0:
            return 0.0
        return float(intersection / union)

    def _nms(self, candidates: List[Tuple[BBox, float]]) -> List[Tuple[BBox, float]]:
        """Applies Non-Maximum Suppression (NMS) on candidates list."""
        if not candidates:
            return []
            
        # Sort by confidence score descending
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
        keep = []
        
        while sorted_candidates:
            best_candidate = sorted_candidates.pop(0)
            keep.append(best_candidate)
            
            # Filter remaining candidates overlapping with best
            sorted_candidates = [
                cand for cand in sorted_candidates
                if self._compute_iou(best_candidate[0], cand[0]) < self.nms_iou_threshold
            ]
            
        return keep

    def detect(self, image: np.ndarray) -> Optional[BBox]:
        """
        Locates the primary face bounding box in BGR image.
        
        Args:
            image: BGR raw image matrix.
            
        Returns:
            Optional[BBox]: BBox containing face coordinates, or None.
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            img_h, img_w = gray.shape
            candidates: List[Tuple[BBox, float]] = []
            
            # Search over multiple scale widths (multi-scale detection)
            scales = [80, 110, 140, 170, 200]
            scales = [s for s in scales if self.min_size <= s <= self.max_size and s < min(img_h, img_w)]
            
            logger.debug(f"Scanning face symmetry over scales: {scales}")
            
            for scale_w in scales:
                scale_h = int(scale_w * self.aspect_ratio)
                
                # Step 1: Symmetry mapping
                sym_scores = self.compute_symmetry_map(gray, scale_w)
                
                # Step 2: Find columns with high symmetry score peaks (> 0.4)
                peaks = np.where(sym_scores > 0.4)[0]
                
                # Sliding vertical window search around symmetry axes
                for xc in peaks:
                    score_sym = float(sym_scores[xc])
                    
                    # Slide y coordinates (face center y search)
                    # Limit search range vertically to prevent overflow
                    y_steps = range(20, img_h - scale_h - 20, max(1, scale_h // 4))
                    for y in y_steps:
                        x = xc - scale_w // 2
                        
                        bbox = BBox(x=x, y=y, w=scale_w, h=scale_h)
                        
                        # Validate boundaries
                        if x < 0 or x + scale_w > img_w or y + scale_h > img_h:
                            continue
                            
                        # Crop candidate gray region
                        crop = gray[y:y+scale_h, x:x+scale_w]
                        if crop.size == 0:
                            continue
                            
                        # Step 3: HOG Template Matching
                        crop_64x64 = cv2.resize(crop, (self.template_dim, self.template_dim))
                        hog_feat = self._extract_hog(crop_64x64)
                        
                        # Cosine similarity: Dot(A, B) / (Norm(A) * Norm(B))
                        dot = np.dot(hog_feat, self.face_template_hog)
                        norm_a = np.linalg.norm(hog_feat)
                        norm_b = np.linalg.norm(self.face_template_hog)
                        score_hog = float(dot / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0
                        
                        # Step 4: Projection Profile Verification
                        score_profile = self.verify_projection_profile(crop)
                        
                        # Step 5: Confidence Score Fusion
                        confidence = float(0.4 * score_sym + 0.4 * score_hog + 0.2 * score_profile)
                        
                        if confidence >= self.confidence_threshold:
                            candidates.append((bbox, confidence))
                            
            # Step 6: Non-Maximum Suppression to filter overlaps
            keep_candidates = self._compute_nms_candidates(candidates)
            
            if keep_candidates:
                best_box, conf = keep_candidates[0]
                logger.info(f"Face detected: {best_box} with confidence: {conf:.4f}")
                return best_box
                
            logger.debug("No valid face candidate matches found.")
            return None
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise e

    def _compute_nms_candidates(self, candidates: List[Tuple[BBox, float]]) -> List[Tuple[BBox, float]]:
        """Calls NMS helper."""
        return self._nms(candidates)
