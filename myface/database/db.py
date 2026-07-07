import os
import json
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import numpy as np

logger = logging.getLogger("myface.database")

class AuditLogEntry(BaseModel):
    """
    Representation of an authentication attempt entry.
    """
    timestamp: str
    liveness_score: float
    similarity_score: float
    authenticated: bool
    status: str

class UserRecord(BaseModel):
    """
    Persisted representation of an enrolled user.
    """
    user_id: str
    username: str
    enrolled_at: str
    # Stored as flat lists, converted to numpy arrays on access
    master_centroid: List[float]
    pca_eigenvectors: List[List[float]]  # Projected eigenspace shape (k, d)
    mean_vector: Optional[List[float]] = None
    audit_logs: List[AuditLogEntry] = Field(default_factory=list)

class Database:
    """
    JSON file-system based database manager for user metadata, biometric templates, and audit logs.
    """
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "db.json")
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Creates the data directory and db.json if they don't exist."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            if not os.path.exists(self.db_path):
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump({}, f)
                logger.info(f"Initialized database file at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database folder: {e}")
            raise IOError(f"Could not access database storage path: {e}")

    def load_user(self, user_id: str) -> Optional[UserRecord]:
        """
        Loads user records from the local DB.
        """
        try:
            if not os.path.exists(self.db_path):
                return None
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            user_data = data.get(user_id)
            if not user_data:
                return None
            
            return UserRecord(**user_data)
        except Exception as e:
            logger.error(f"Failed to load user {user_id}: {e}")
            return None

    def save_user(self, user: UserRecord) -> bool:
        """
        Persists user record.
        """
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            data[user.user_id] = user.model_dump()
            
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Successfully saved user {user.username} (ID: {user.user_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to save user {user.user_id}: {e}")
            return False

    def delete_user(self, user_id: str) -> bool:
        """
        Removes user record.
        """
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if user_id in data:
                del data[user_id]
                with open(self.db_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Successfully deleted user ID: {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False

    def list_users(self) -> List[Dict[str, str]]:
        """
        Lists all usernames and ids stored in the database.
        """
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [{"user_id": uid, "username": info["username"]} for uid, info in data.items()]
        except Exception as e:
            logger.error(f"Failed to list database users: {e}")
            return []

# Singleton dependency helper
_db_instance: Optional[Database] = None

def get_db(data_dir: str) -> Database:
    """Dependency injector pattern to obtain database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(data_dir=data_dir)
    return _db_instance
