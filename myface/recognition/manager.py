import logging
import uuid
import datetime
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from myface.database.db import Database, UserRecord, AuditLogEntry
from myface.dataset.manager import DatasetManager
from myface.detection.detector import FaceDetector
from myface.alignment.aligner import FaceAligner
from myface.feature_extraction.extractor import FeatureExtractor
from myface.anti_spoof.liveness import LivenessDetector
from myface.matching.matcher import BiometricMatcher

logger = logging.getLogger("myface.recognition")

class BiometricOrchestrator:
    """
    Main manager orchestrating the biometric pipelines.
    
    Coordinates camera acquisition/files mapping, face localization, 
    eye-alignment warping, anti-spoof checks, eigenspace calculations,
    similarity matching, and result auditing writes.
    """
    def __init__(
        self,
        db: Database,
        dataset_mgr: DatasetManager,
        detector: FaceDetector,
        aligner: FaceAligner,
        extractor: FeatureExtractor,
        liveness: LivenessDetector,
        matcher: BiometricMatcher
    ):
        self.db = db
        self.dataset_mgr = dataset_mgr
        self.detector = detector
        self.aligner = aligner
        self.extractor = extractor
        self.liveness = liveness
        self.matcher = matcher

    def enroll(self, username: str, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Runs full enrollment registration workflow.
        
        Saves raw assets, aligns crops, runs PCA-SVD calculations, 
        and writes records to Database.
        """
        try:
            from myface.training.enroll import EnrollmentPipeline
            pipeline = EnrollmentPipeline(self.detector, self.aligner, self.extractor)
            
            # Execute processing
            enroll_data = pipeline.process_enrollment(images, username)
            
            user_id = str(uuid.uuid4())
            now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
            # Save raw and aligned crops cache on disk
            for idx, raw_img in enumerate(images):
                self.dataset_mgr.save_enrollment_snapshot(username, idx, raw_img, is_aligned=False)
            for idx, crop in enumerate(enroll_data["aligned_crops"]):
                self.dataset_mgr.save_enrollment_snapshot(username, idx, crop, is_aligned=True)

            user_record = UserRecord(
                user_id=user_id,
                username=username,
                enrolled_at=now_str,
                master_centroid=enroll_data["master_centroid"],
                pca_eigenvectors=enroll_data["pca_eigenvectors"],
                mean_vector=enroll_data["mean_vector"],
                audit_logs=[]
            )
            
            success = self.db.save_user(user_record)
            if not success:
                raise RuntimeError("Failed to persist user profile into database storage.")
                
            return {
                "success": True,
                "user_id": user_id,
                "username": username,
                "enrolled_at": now_str
            }
        except Exception as e:
            logger.error(f"Enrollment pipeline execution failed: {e}")
            return {"success": False, "error": str(e)}

    def authenticate(
        self,
        user_id: str,
        image: np.ndarray,
        match_thresh: float,
        liveness_thresh: float
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs authentication query pipeline.
        
        Args:
            user_id: Targeted database user profile.
            image: Query input image matrix.
            match_thresh: Validation score threshold.
            liveness_thresh: Liveness validation threshold.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (authenticated_status, payload)
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status_msg = "Unknown"
        liveness_score = 0.0
        similarity_score = 0.0
        authenticated = False

        try:
            user = self.db.load_user(user_id)
            if not user:
                status_msg = "User record not found."
                return False, {"authenticated": False, "status": status_msg}

            # 1. Face Detection
            bbox = self.detector.detect(image)
            if bbox is None:
                status_msg = "No face detected in query frame."
                return False, {"authenticated": False, "status": status_msg}

            # 2. Geometry Alignment
            crop = self.aligner.align(image, bbox)

            # 3. Liveness Check
            is_live, liveness_score = self.liveness.analyze_liveness(crop, liveness_thresh)
            if not is_live:
                status_msg = "Liveness check failed (Spoof attempt blocked)."
                self._log_attempt(user, now_str, liveness_score, 0.0, False, status_msg)
                return False, {"authenticated": False, "status": status_msg, "liveness_score": liveness_score}

            # 4. Feature Projection & Matching
            eigenvectors = np.array(user.pca_eigenvectors)
            # Reconstruct mean vector (assuming average vector is stored or mock computed here)
            # For boilerplate, extract query coordinates
            # TODO: Match logic calculations
            query_raw = self.extractor.extract_features(crop)
            
            # Extract mean vector from DB, fallback to zeros if missing
            if user.mean_vector is not None:
                mean_vector = np.array(user.mean_vector)
            else:
                mean_vector = np.zeros_like(query_raw)
                
            query_projected = self.extractor.project_pca(query_raw, eigenvectors, mean_vector)
            
            centroid = np.array(user.master_centroid)
            
            # Run comparison verify
            authenticated, similarity_score = self.matcher.verify(query_projected, centroid, match_thresh)
            status_msg = "Authentication Succeeded" if authenticated else "Match score below threshold."

            self._log_attempt(user, now_str, liveness_score, similarity_score, authenticated, status_msg)
            
            return authenticated, {
                "authenticated": authenticated,
                "status": status_msg,
                "similarity_score": similarity_score,
                "liveness_score": liveness_score
            }
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False, {"authenticated": False, "status": f"System error: {e}"}

    def _log_attempt(
        self,
        user: UserRecord,
        timestamp: str,
        liveness_score: float,
        similarity_score: float,
        authenticated: bool,
        status: str
    ) -> None:
        """Appends log entry and saves database user state."""
        try:
            entry = AuditLogEntry(
                timestamp=timestamp,
                liveness_score=liveness_score,
                similarity_score=similarity_score,
                authenticated=authenticated,
                status=status
            )
            # Limit logs history to last 50 attempts
            user.audit_logs.append(entry)
            if len(user.audit_logs) > 50:
                user.audit_logs.pop(0)
            self.db.save_user(user)
        except Exception as e:
            logger.error(f"Failed to record audit attempt log: {e}")
