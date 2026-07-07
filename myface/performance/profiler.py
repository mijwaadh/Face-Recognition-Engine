import time
import logging
from typing import Optional

logger = logging.getLogger("myface.performance")

class CodeTimer:
    """
    Context manager to track execution latency of blocks of code.
    """
    def __init__(self, description: str):
        self.description = description
        self.start_time: Optional[float] = None

    def __enter__(self) -> "CodeTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            elapsed = (time.perf_counter() - self.start_time) * 1000.0  # in ms
            logger.info(f"PERFORMANCE: Block [{self.description}] executed in {elapsed:.2f} ms")
