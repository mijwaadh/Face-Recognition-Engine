import logging
from typing import Tuple, Optional, List
import numpy as np

logger = logging.getLogger("myface.matching")

class BiometricMatcher:
    """
    Computes biometric similarity scores and distances between query vectors and enrolled templates.
    
    Mathematical Principles:
    1. Cosine Similarity:
       Measures the cosine of the angle between two multi-dimensional vectors:
         Sim(u, v) = (u . v) / (||u|| * ||v||)
    2. Chi-Square Distance:
       Compares bin frequencies of LBP histograms:
         D_chi(p, q) = Sum_i ( (p_i - q_i)^2 / (p_i + q_i + epsilon) )
    3. Mahalanobis Distance:
       Measures distance in PCA eigenspace taking coordinate variances into account:
         D_mahal(x, y) = sqrt( (x - y)^T * Inv(Sigma) * (x - y) )
       For diagonal covariance (variance vector), it maps to:
         D_mahal(x, y) = sqrt( Sum_i ( (x_i - y_i)^2 / (sigma_i^2 + epsilon) ) )
    4. Confidence Estimation:
       Converts the distance/similarity difference to a confidence metric using a Sigmoid function:
         Conf = 1 / (1 + exp(-k * (similarity - threshold)))
    5. Threshold Calibration:
       Sorts imposter scores descending, and identifies the score index matching 
       the target FAR ratio to find the optimal decision boundary.
    """
    def compute_cosine_similarity(self, u: np.ndarray, v: np.ndarray) -> float:
        """
        Computes the Cosine similarity between two feature vectors.
        """
        try:
            dot_product = np.dot(u, v)
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0:
                return 0.0
            return float(dot_product / (norm_u * norm_v))
        except Exception as e:
            logger.error(f"Cosine similarity error: {e}")
            return 0.0

    def compute_chi_square_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Computes Chi-Square distance between LBP histogram bins.
        """
        try:
            epsilon = 1e-10
            diff = (p - q) ** 2
            summ = p + q + epsilon
            return float(np.sum(diff / summ))
        except Exception as e:
            logger.error(f"Chi-square calculation error: {e}")
            return float("inf")

    def compute_mahalanobis_distance(
        self,
        query: np.ndarray,
        centroid: np.ndarray,
        cov_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes Mahalanobis distance inside the PCA Eigenspace.
        
        Args:
            query: Lower dimensional query coordinate projection.
            centroid: Enrolled master template centroid projection.
            cov_matrix: Optional covariance matrix of projections.
            
        Returns:
            float: Mahalanobis distance.
        """
        try:
            diff = query - centroid
            if cov_matrix is not None:
                if len(cov_matrix.shape) == 1:
                    # Diagonal covariance (variance vector) representation
                    dist = np.sqrt(np.sum((diff ** 2) / (cov_matrix + 1e-6)))
                else:
                    # Full covariance matrix representation
                    # Add small regularization to diagonal to ensure invertibility
                    reg_cov = cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-6
                    inv_cov = np.linalg.inv(reg_cov)
                    dist = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
                return float(dist)
            
            # Fallback to standard Euclidean distance
            return float(np.linalg.norm(diff))
        except Exception as e:
            logger.error(f"Mahalanobis computation failed: {e}")
            raise e

    def compute_confidence_score(self, similarity: float, threshold: float) -> float:
        """
        Computes biometric match confidence using a Sigmoid mapping around the threshold.
        """
        try:
            # Steepness scaling factor k
            k = 15.0
            logit = k * (similarity - threshold)
            confidence = float(1.0 / (1.0 + np.exp(-logit)))
            return confidence
        except Exception as e:
            logger.warning(f"Confidence score calculation failed: {e}")
            return 0.5

    def calibrate_threshold(self, genuine_scores: List[float], imposter_scores: List[float], target_far: float = 0.01) -> float:
        """
        Calibrates the verification threshold to meet a target False Acceptance Rate (FAR).
        
        Args:
            genuine_scores: Similarity scores of correct users.
            imposter_scores: Similarity scores of imposters/spoof users.
            target_far: Desired False Acceptance Rate ratio (e.g. 0.01 = 1% FAR).
            
        Returns:
            float: Calibrated threshold value.
        """
        try:
            if not imposter_scores:
                return 0.5
            sorted_imposters = sorted(imposter_scores, reverse=True)
            idx = int(np.floor(target_far * len(sorted_imposters)))
            idx = max(0, min(idx, len(sorted_imposters) - 1))
            calibrated_thresh = float(sorted_imposters[idx])
            logger.info(f"Threshold calibrated to {calibrated_thresh:.4f} for target FAR={target_far}")
            return calibrated_thresh
        except Exception as e:
            logger.error(f"Threshold calibration failed: {e}")
            return 0.5

    def verify(self, query_vector: np.ndarray, master_centroid: np.ndarray, threshold: float) -> Tuple[bool, float]:
        """
        Verifies if query vector is matching the master centroid.
        
        Args:
            query_vector: Feature vector of the input query face.
            master_centroid: Enrolled master template vector.
            threshold: Match similarity validation threshold.
            
        Returns:
            Tuple[bool, float]: (is_match, similarity_score)
        """
        try:
            # Similarity = Cosine similarity (1 is identical, -1 is opposite)
            sim_score = self.compute_cosine_similarity(query_vector, master_centroid)
            
            # Map cosine range [-1.0, 1.0] to [0.0, 1.0]
            normalized_score = float((sim_score + 1.0) / 2.0)
            
            is_match = normalized_score >= threshold
            logger.info(f"Verification decision: {is_match} (Score: {normalized_score:.4f}, Threshold: {threshold:.4f})")
            
            return is_match, normalized_score
        except Exception as e:
            logger.error(f"Verification pipeline failed: {e}")
            raise e
