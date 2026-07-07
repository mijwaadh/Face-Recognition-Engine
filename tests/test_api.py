import cv2
import io
import pytest
import numpy as np
from fastapi.testclient import TestClient

def create_dummy_face_image_bytes(width: int = 200, height: int = 200) -> bytes:
    """Generates a dummy 3-channel image using NumPy and returns its encoded PNG bytes."""
    # Create black canvas
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Draw simple facial bounding proportions (circle for head, smaller circles for eyes)
    cv2.circle(img, (100, 100), 80, (255, 255, 255), -1) # Face oval
    cv2.circle(img, (75, 75), 10, (0, 0, 0), -1)       # Left eye
    cv2.circle(img, (125, 75), 10, (0, 0, 0), -1)      # Right eye
    cv2.rectangle(img, (80, 130), (120, 140), (0, 0, 0), -1) # Mouth
    
    # Encode as PNG bytes
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode dummy matrix to PNG bytes.")
    return buffer.tobytes()

def test_health_check(client: TestClient):
    """Verifies that the diagnostics check returns status 200."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["debug"] is True

def test_enrollment_validation(client: TestClient):
    """Verifies that registration fails with bad payloads or too few files."""
    # Test missing parameters
    response = client.post("/api/v1/enroll", data={})
    assert response.status_code == 422

    # Test less than 3 files
    img_bytes = create_dummy_face_image_bytes()
    files = [
        ("files", ("face1.png", img_bytes, "image/png")),
        ("files", ("face2.png", img_bytes, "image/png"))
    ]
    response = client.post("/api/v1/enroll", data={"username": "alice"}, files=files)
    assert response.status_code == 400
    assert "at least 3" in response.json()["detail"]

def test_enroll_and_authenticate_and_management(client: TestClient):
    """Runs a full lifecycle test: enroll, list profiles, authenticate, and delete."""
    img_bytes = create_dummy_face_image_bytes()
    
    # 1. Enroll User (Provide 3 dummy snapshots)
    enroll_files = [
        ("files", ("face1.png", img_bytes, "image/png")),
        ("files", ("face2.png", img_bytes, "image/png")),
        ("files", ("face3.png", img_bytes, "image/png"))
    ]
    enroll_resp = client.post("/api/v1/enroll", data={"username": "bob"}, files=enroll_files)
    # Face detection might return None on dummy drawing depending on symmetry math,
    # but the orchestrator mock will register bob successfully.
    assert enroll_resp.status_code == 201
    enroll_data = enroll_resp.json()
    assert enroll_data["success"] is True
    assert enroll_data["username"] == "bob"
    user_id = enroll_data["user_id"]
    assert user_id is not None

    # 2. List Profiles
    list_resp = client.get("/api/v1/users")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert len(list_data["users"]) == 1
    assert list_data["users"][0]["username"] == "bob"

    # 3. Authenticate User (Expected pass/fail depending on similarity score)
    auth_file = {"file": ("query.png", img_bytes, "image/png")}
    auth_resp = client.post("/api/v1/authenticate", data={"user_id": user_id}, files=auth_file)
    assert auth_resp.status_code == 200
    auth_data = auth_resp.json()
    # auth_data will contain authenticated, status, similarity_score, and liveness_score
    assert "authenticated" in auth_data
    assert "liveness_score" in auth_data

    # 4. Verify Logs and Metrics Endpoints
    logs_resp = client.get("/api/v1/logs")
    assert logs_resp.status_code == 200
    assert "logs" in logs_resp.json()

    metrics_resp = client.get("/api/v1/metrics")
    assert metrics_resp.status_code == 200
    assert "eer" in metrics_resp.json()
    assert "auc" in metrics_resp.json()

    # 5. Serve Frontend UI Check
    ui_resp = client.get("/")
    assert ui_resp.status_code == 200
    assert "text/html" in ui_resp.headers["content-type"]

    # 6. Delete User Profile
    del_resp = client.delete(f"/api/v1/users/{user_id}")
    assert del_resp.status_code == 204

    # Verify deleted
    list_resp2 = client.get("/api/v1/users")
    assert len(list_resp2.json()["users"]) == 0
