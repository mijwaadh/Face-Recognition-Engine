"""
Training module.
Aggregates multi-image snapshots and trains personalized Eigenspaces (SVD).
"""

from myface.training.enroll import EnrollmentPipeline

__all__ = ["EnrollmentPipeline"]
