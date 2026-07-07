from fastapi import Depends
from functools import lru_cache

from myface.configuration.config import Settings, get_settings
from myface.database.db import Database, get_db
from myface.dataset.manager import DatasetManager
from myface.detection.detector import FaceDetector
from myface.alignment.aligner import FaceAligner
from myface.feature_extraction.extractor import FeatureExtractor
from myface.anti_spoof.liveness import LivenessDetector
from myface.matching.matcher import BiometricMatcher
from myface.recognition.manager import BiometricOrchestrator
from myface.quality_assessment.assessment import FrameQualityEvaluator
from myface.camera.capture import CameraManager

_camera_manager_instance = None

def get_camera_manager() -> CameraManager:
    global _camera_manager_instance
    if _camera_manager_instance is None:
        _camera_manager_instance = CameraManager()
    return _camera_manager_instance

@lru_cache()
def get_quality_evaluator() -> FrameQualityEvaluator:
    return FrameQualityEvaluator()

def get_dataset_manager(settings: Settings = Depends(get_settings)) -> DatasetManager:
    return DatasetManager(data_dir=settings.data_dir)

@lru_cache()
def get_face_detector() -> FaceDetector:
    return FaceDetector()

@lru_cache()
def get_face_aligner() -> FaceAligner:
    return FaceAligner()

@lru_cache()
def get_feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()

@lru_cache()
def get_liveness_detector() -> LivenessDetector:
    return LivenessDetector()

@lru_cache()
def get_biometric_matcher() -> BiometricMatcher:
    return BiometricMatcher()

def get_orchestrator(
    settings: Settings = Depends(get_settings),
    dataset_mgr: DatasetManager = Depends(get_dataset_manager),
    detector: FaceDetector = Depends(get_face_detector),
    aligner: FaceAligner = Depends(get_face_aligner),
    extractor: FeatureExtractor = Depends(get_feature_extractor),
    liveness: LivenessDetector = Depends(get_liveness_detector),
    matcher: BiometricMatcher = Depends(get_biometric_matcher)
) -> BiometricOrchestrator:
    db = get_db(data_dir=settings.data_dir)
    return BiometricOrchestrator(
        db=db,
        dataset_mgr=dataset_mgr,
        detector=detector,
        aligner=aligner,
        extractor=extractor,
        liveness=liveness,
        matcher=matcher
    )
