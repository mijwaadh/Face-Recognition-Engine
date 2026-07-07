import pytest
import os
import shutil
from typing import Generator, Optional, Any
import numpy as np
from fastapi.testclient import TestClient

from myface.configuration.config import Settings, get_settings
from myface.api.main import app
from myface.database.db import get_db
from myface.api.dependencies import get_face_detector, get_quality_evaluator
from myface.detection.detector import FaceDetector, BBox
from myface.quality_assessment.assessment import FrameQualityEvaluator, QualityReport

class MockFaceDetector(FaceDetector):
    def detect(self, image: np.ndarray) -> Optional[BBox]:
        # Return a valid bounding box covering the canvas to allow alignment to proceed
        return BBox(x=0, y=0, w=image.shape[1], h=image.shape[0])

class MockFrameQualityEvaluator(FrameQualityEvaluator):
    def evaluate_frame(self, image: np.ndarray, bbox: Optional[Any] = None) -> QualityReport:
        return QualityReport(
            overall_score=1.0,
            blur_score=1.0,
            focus_score=1.0,
            brightness_score=1.0,
            contrast_score=1.0,
            noise_score=1.0,
            exposure_score=1.0,
            face_size_score=1.0,
            pose_score=1.0,
            occlusion_score=1.0,
            passed=True,
            rejection_reasons=[]
        )

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> Generator[str, None, None]:
    """Provides a temporary directory path for test databases and cache files."""
    tmp_dir = tmp_path_factory.mktemp("face_auth_test_data")
    yield str(tmp_dir)
    # Cleanup temp directory after testing session finishes
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

@pytest.fixture(scope="session", autouse=True)
def settings_override(test_data_dir: str) -> Settings:
    """Overrides default settings to point paths to temp test directories."""
    test_settings = Settings(
        API_HOST="127.0.0.1",
        API_PORT=8001,
        DEBUG=True,
        DATA_DIR=test_data_dir,
        LOG_DIR=os.path.join(test_data_dir, "logs"),
        MATCH_THRESHOLD=0.75,
        LIVENESS_THRESHOLD=0.85
    )
    
    # Override settings and detector generator functions
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_face_detector] = lambda: MockFaceDetector()
    app.dependency_overrides[get_quality_evaluator] = lambda: MockFrameQualityEvaluator()
    return test_settings

@pytest.fixture
def client(settings_override: Settings) -> Generator[TestClient, None, None]:
    """Returns a TestClient instance bound to the overridden app configurations."""
    # Ensure database is clean before each test run
    db = get_db(settings_override.data_dir)
    # Clear user list entries
    for u in db.list_users():
        db.delete_user(u["user_id"])
        
    with TestClient(app) as test_client:
        yield test_client
