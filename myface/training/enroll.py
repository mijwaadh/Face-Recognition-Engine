import logging
from typing import List, Tuple, Dict, Any
import numpy as np
from myface.detection.detector import FaceDetector
from myface.alignment.aligner import FaceAligner
from myface.feature_extraction.extractor import FeatureExtractor

logger = logging.getLogger("myface.training")

class EnrollmentPipeline:
    """
    Assembles multi-image face acquisitions into a personalized PCA Eigenspace.
    """
    def __init__(self, detector: FaceDetector, aligner: FaceAligner, extractor: FeatureExtractor):
        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor

    def train_eigenspace(self, raw_features: List[np.ndarray], k_components: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Singular Value Decomposition (SVD) on input vectors to compile eigenvectors.
        
        Args:
            raw_features: List of combined high-dimensional vectors (HOG+LBP).
            k_components: Number of principal components to preserve.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (eigenvectors, mean_vector)
        """
        try:
            # Convert list to data matrix X shape (n, d)
            X = np.stack(raw_features, axis=0)
            
            # TODO: Center columns: X_centered = X - mean
            mean_vector = np.mean(X, axis=0)
            X_centered = X - mean_vector
            
            # SVD decomposition: U, S, Vt = svd(X_centered)
            # Principal components are the columns of V (rows of Vt)
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Select top k rows of Vt
            eigenvectors = Vt[:k_components, :]
            
            logger.info(f"Generated Eigenspace eigenvectors of shape {eigenvectors.shape}")
            return eigenvectors, mean_vector
        except Exception as e:
            logger.error(f"Eigenspace compilation failed: {e}")
            raise e

    def process_enrollment(self, images: List[np.ndarray], username: str) -> Dict[str, Any]:
        """
        Executes registration pipeline.
        
        Args:
            images: Raw enrollment snapshots.
            username: User name registering.
            
        Returns:
            Dict[str, Any]: Compiled data payload containing 'centroid', 'eigenvectors', and 'mean'.
        """
        aligned_crops = []
        raw_features = []

        logger.info(f"Starting enrollment pipeline for user: {username} with {len(images)} images.")
        
        for idx, img in enumerate(images):
            bbox = self.detector.detect(img)
            if bbox is None:
                logger.warning(f"Face detection missed on snapshot idx: {idx}. Skipping.")
                continue

            crop = self.aligner.align(img, bbox)
            aligned_crops.append(crop)
            
            # Extract raw spatial concatenated descriptors
            features = self.extractor.extract_features(crop)
            raw_features.append(features)

        if len(raw_features) < 3:
            logger.error("Enrollment failed: Insufficient face samples acquired.")
            raise ValueError("At least 3 valid face snapshots must be captured for successful enrollment.")

        # Train Eigenspace SVD
        eigenvectors, mean_vector = self.train_eigenspace(raw_features)
        
        # Project raw features into new PCA space and compute centroid
        projected = [self.extractor.project_pca(f, eigenvectors, mean_vector) for f in raw_features]
        master_centroid = np.mean(projected, axis=0)
        
        logger.info(f"Successfully compiled registration centroids for {username}.")
        
        return {
            "master_centroid": master_centroid.tolist(),
            "pca_eigenvectors": eigenvectors.tolist(),
            "mean_vector": mean_vector.tolist(),
            "aligned_crops": aligned_crops
        }
