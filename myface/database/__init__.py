"""
Database module.
Handles persistent storage of user credentials, face templates, and audit logs.
"""

from myface.database.db import Database, UserRecord, get_db

__all__ = ["Database", "UserRecord", "get_db"]
