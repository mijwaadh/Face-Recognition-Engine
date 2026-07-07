"""
Performance module.
Exposes context managers and decorators to monitor processing latency.
"""

from myface.performance.profiler import CodeTimer

__all__ = ["CodeTimer"]
