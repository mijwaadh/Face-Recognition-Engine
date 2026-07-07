from pydantic import BaseModel, Field
from typing import List, Optional

class HealthResponse(BaseModel):
    status: str
    debug: bool

class EnrollmentResponse(BaseModel):
    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    enrolled_at: Optional[str] = None
    error: Optional[str] = None

class AuthenticateResponse(BaseModel):
    authenticated: bool
    status: str
    similarity_score: Optional[float] = None
    liveness_score: Optional[float] = None

class LogEntryResponse(BaseModel):
    timestamp: str
    liveness_score: float
    similarity_score: float
    authenticated: bool
    status: str

class UserProfileResponse(BaseModel):
    user_id: str
    username: str
    enrolled_at: str
    audit_logs: List[LogEntryResponse]

class UserListResponse(BaseModel):
    users: List[UserProfileResponse]
