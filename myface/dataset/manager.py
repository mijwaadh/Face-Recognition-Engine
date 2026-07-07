import os
import cv2
import json
import time
import datetime
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("myface.dataset")

class DatasetManager:
    """
    Manages enrollment dataset collection, validation, and storage logic.
    
    Provisions dataset directories (raw, processed, rejected, enrollment, validation, spoof),
    validates incoming frames for quality and duplicate presence, computes metadata, 
    and drives automated image collection workflows.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
        # Configure target subfolders
        self.base_dataset_dir = os.path.join(data_dir, "dataset")
        self.raw_dir = os.path.join(self.base_dataset_dir, "raw")
        self.processed_dir = os.path.join(self.base_dataset_dir, "processed")
        self.rejected_dir = os.path.join(self.base_dataset_dir, "rejected")
        self.enrollment_dir = os.path.join(self.base_dataset_dir, "enrollment")
        self.validation_dir = os.path.join(self.base_dataset_dir, "validation")
        self.spoof_dir = os.path.join(self.base_dataset_dir, "spoof")
        
        self._initialize_directories()

    def _initialize_directories(self) -> None:
        """Creates required directory structures if missing."""
        try:
            for folder in [
                self.raw_dir, self.processed_dir, self.rejected_dir,
                self.enrollment_dir, self.validation_dir, self.spoof_dir
            ]:
                os.makedirs(folder, exist_ok=True)
            logger.info(f"Initialized dataset directories under: {self.base_dataset_dir}")
        except Exception as e:
            logger.error(f"Failed to create dataset directories: {e}")
            raise IOError(f"Could not provision dataset folders: {e}")

    def compute_quality_metrics(self, gray_image: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Evaluates image quality metrics: sharpness, brightness, and contrast.
        
        Returns:
            Tuple[float, float, float, float]: (quality_score, sharpness, brightness, contrast)
        """
        # 1. Sharpness (Laplacian variance)
        sharpness = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
        
        # 2. Brightness (Average intensity)
        brightness = float(np.mean(gray_image))
        
        # 3. Contrast (Standard deviation of intensity)
        contrast = float(np.std(gray_image))
        
        # Normalize sub-scores to [0.0, 1.0] range
        sharp_score = min(1.0, sharpness / 400.0)
        
        # Ideal brightness is around middle gray (127.0). Penalize extreme dark/light
        bright_score = 1.0 - (abs(brightness - 127.0) / 127.0)
        
        # Ideal contrast std >= 45.0
        contrast_score = min(1.0, contrast / 45.0)
        
        # Linear weighted score fusion
        quality_score = float((0.5 * sharp_score) + (0.3 * bright_score) + (0.2 * contrast_score))
        quality_score = float(np.clip(quality_score, 0.0, 1.0))
        
        return quality_score, sharpness, brightness, contrast

    def is_duplicate(
        self, 
        gray_image: np.ndarray, 
        existing_images: List[np.ndarray], 
        correlation_threshold: float = 0.98,
        mae_threshold: float = 2.0
    ) -> bool:
        """
        Determines if the image is a duplicate of any image in the list.
        
        Uses histogram correlation and Mean Absolute Error (MAE) comparisons.
        """
        if not existing_images:
            return False
            
        # Calculate normalized histogram for current image
        hist_curr = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        cv2.normalize(hist_curr, hist_curr, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        for ref_img in existing_images:
            if ref_img.shape != gray_image.shape:
                continue
                
            # Histogram comparison
            hist_ref = cv2.calcHist([ref_img], [0], None, [256], [0, 256])
            cv2.normalize(hist_ref, hist_ref, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            corr = cv2.compareHist(hist_curr, hist_ref, cv2.HISTCMP_CORREL)
            
            # Mean Absolute Error (MAE)
            mae = float(np.mean(np.abs(gray_image.astype(np.int32) - ref_img.astype(np.int32))))
            
            if corr >= correlation_threshold or mae <= mae_threshold:
                logger.warning(f"Duplicate frame detected (Hist correlation: {corr:.4f}, MAE: {mae:.2f})")
                return True
                
        return False

    def estimate_pose(self, gray_face: np.ndarray) -> str:
        """
        Estimates face yaw orientation profile using intensity centroid asymmetry.
        
        If the face is turned right, one side has a different perspective/shading 
        profile than the other. Calculates horizontal offset of mass center.
        """
        try:
            h, w = gray_face.shape
            # Moments of gray image
            M = cv2.moments(gray_face)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                # Normalised offset from central vertical line [-1.0, 1.0]
                offset_x = (cx - (w / 2.0)) / (w / 2.0)
                
                # Threshold bounds for profile yaw estimation
                if offset_x > 0.05:
                    return "Right Profile"
                elif offset_x < -0.05:
                    return "Left Profile"
            return "Frontal"
        except Exception as e:
            logger.debug(f"Pose estimation calculation failed: {e}")
            return "Unknown"

    def save_image_with_metadata(
        self,
        image: np.ndarray,
        user_id: str,
        category: str,
        pose: str,
        quality_score: float,
        brightness: float,
        contrast: float,
        index: int
    ) -> Dict[str, Any]:
        """
        Saves the image matrix to the corresponding category folder and writes metadata log.
        
        Args:
            image: BGR or Grayscale image matrix.
            user_id: Enrolling user ID.
            category: Folder category ('raw', 'processed', 'rejected', 'enrollment', 'validation', 'spoof').
            pose: Estimated face pose description.
            quality_score: Computed image quality.
            brightness: Average luminance.
            contrast: Contrast std.
            index: Frame index serial.
            
        Returns:
            Dict[str, Any]: Saved image metadata dictionary.
        """
        folder_map = {
            "raw": self.raw_dir,
            "processed": self.processed_dir,
            "rejected": self.rejected_dir,
            "enrollment": self.enrollment_dir,
            "validation": self.validation_dir,
            "spoof": self.spoof_dir
        }
        
        target_dir = folder_map.get(category.lower(), self.base_dataset_dir)
        filename = f"{user_id}_{category}_{index:03d}_{pose.replace(' ', '_').lower()}.png"
        file_path = os.path.join(target_dir, filename)
        
        try:
            cv2.imwrite(file_path, image)
            logger.debug(f"Saved image to {file_path}")
        except Exception as e:
            logger.error(f"Failed to write image {file_path}: {e}")
            raise e

        # Compile metadata
        metadata = {
            "user_id": user_id,
            "filename": filename,
            "category": category,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "image_quality": round(quality_score, 4),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "pose_estimation": pose
        }
        
        # Save metadata to JSON log adjacent to target file
        meta_path = file_path + ".json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write metadata log {meta_path}: {e}")
            
        return metadata

    def collect_enrollment_batch(
        self,
        user_id: str,
        frames: List[np.ndarray],
        blur_threshold: float = 80.0,
        quality_threshold: float = 0.5,
        capture_interval: float = 0.3
    ) -> Dict[str, Any]:
        """
        Process and validate a batch of enrollment images.
        
        Filters for blur, duplicates, and minimum quality thresholds, estimates pose,
        and saves outputs into enrollment, raw, processed, or rejected folders.
        
        Args:
            user_id: Targeted user ID.
            frames: Captured raw BGR snapshots.
            blur_threshold: Minimum Laplacian variance.
            quality_threshold: Minimum combined quality score.
            capture_interval: Minimum mock time spacing between valid frames.
            
        Returns:
            Dict[str, Any]: Statistics report of the collection session.
        """
        logger.info(f"Starting batch validation for user {user_id} with {len(frames)} source frames.")
        
        accepted_metadata = []
        rejected_metadata = []
        accepted_gray_images = []
        
        last_captured_time = 0.0
        
        for idx, frame in enumerate(frames):
            now = time.time()
            # Enforce capture interval pacing
            if now - last_captured_time < capture_interval:
                time.sleep(capture_interval - (now - last_captured_time))
            
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.error(f"Failed to convert frame index {idx} to gray: {e}")
                continue

            # Compute quality indicators
            q_score, sharpness, brightness, contrast = self.compute_quality_metrics(gray)
            pose = self.estimate_pose(gray)
            
            # Determine validation failures
            reasons = []
            if sharpness < blur_threshold:
                reasons.append(f"Blurry (Sharpness {sharpness:.1f} < {blur_threshold})")
            if q_score < quality_threshold:
                reasons.append(f"Low Quality (Score {q_score:.2f} < {quality_threshold})")
            if self.is_duplicate(gray, accepted_gray_images):
                reasons.append("Duplicate frame")
                
            if reasons:
                # Save to rejected folder
                reason_str = ", ".join(reasons)
                logger.warning(f"Frame index {idx} rejected: {reason_str}")
                meta = self.save_image_with_metadata(
                    image=frame,
                    user_id=user_id,
                    category="rejected",
                    pose=pose,
                    quality_score=q_score,
                    brightness=brightness,
                    contrast=contrast,
                    index=idx
                )
                meta["rejection_reasons"] = reasons
                rejected_metadata.append(meta)
            else:
                # Save to raw, processed, and enrollment
                logger.info(f"Frame index {idx} accepted with quality: {q_score:.4f} (Pose: {pose})")
                
                # Raw representation
                self.save_image_with_metadata(
                    image=frame,
                    user_id=user_id,
                    category="raw",
                    pose=pose,
                    quality_score=q_score,
                    brightness=brightness,
                    contrast=contrast,
                    index=idx
                )
                
                # Processed crop representation (assuming whole image here for skeleton boundaries,
                # downstream alignment saves aligned crop)
                self.save_image_with_metadata(
                    image=frame,
                    user_id=user_id,
                    category="processed",
                    pose=pose,
                    quality_score=q_score,
                    brightness=brightness,
                    contrast=contrast,
                    index=idx
                )
                
                # Enrollment final selection
                meta = self.save_image_with_metadata(
                    image=frame,
                    user_id=user_id,
                    category="enrollment",
                    pose=pose,
                    quality_score=q_score,
                    brightness=brightness,
                    contrast=contrast,
                    index=idx
                )
                accepted_metadata.append(meta)
                accepted_gray_images.append(gray)
                last_captured_time = time.time()
                
        return {
            "user_id": user_id,
            "total_processed": len(frames),
            "total_accepted": len(accepted_metadata),
            "total_rejected": len(rejected_metadata),
            "accepted_files": [m["filename"] for m in accepted_metadata],
            "rejected_files": [m["filename"] for m in rejected_metadata]
        }

    def save_enrollment_snapshot(self, username: str, index: int, image: np.ndarray, is_aligned: bool = False) -> str:
        """
        Saves a single snapshot matrix to raw/processed or enrollment subfolders for backward compatibility.
        """
        category = "processed" if is_aligned else "raw"
        meta = self.save_image_with_metadata(
            image=image,
            user_id=username,
            category=category,
            pose="frontal",
            quality_score=1.0,
            brightness=127.0,
            contrast=50.0,
            index=index
        )
        folder_map = {
            "raw": self.raw_dir,
            "processed": self.processed_dir
        }
        filename = meta["filename"]
        return os.path.join(folder_map[category], filename)

    def load_aligned_images(self, username: str) -> List[np.ndarray]:
        """
        Loads all processed/aligned png files stored for a user from their folder.
        """
        loaded_images = []
        try:
            for file in os.listdir(self.processed_dir):
                if file.startswith(username) and file.endswith(".png"):
                    path = os.path.join(self.processed_dir, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        loaded_images.append(img)
            logger.info(f"Loaded {len(loaded_images)} aligned grayscale images for {username}")
        except Exception as e:
            logger.error(f"Error reading dataset files for {username}: {e}")
        return loaded_images
