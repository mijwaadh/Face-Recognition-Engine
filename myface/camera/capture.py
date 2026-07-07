import cv2
import time
import threading
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np

logger = logging.getLogger("myface.camera")

class CameraStream:
    """
    Manages a single hardware camera video capture stream.
    
    Spawns a dedicated background thread to continuously pull frames, 
    tracks capture latency and FPS performance, and automatically 
    reconnects if the connection drops.
    """
    def __init__(self, camera_idx: int, width: int = 640, height: int = 480, target_fps: int = 30):
        self.camera_idx = camera_idx
        self.width = width
        self.height = height
        self.target_fps = target_fps
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.last_timestamp: float = 0.0
        self.is_connected: bool = False
        
        # Thread control
        self.running: bool = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        
        # Health & Performance Metrics
        self.frame_count: int = 0
        self.dropped_frame_count: int = 0
        self.reconnect_count: int = 0
        self.read_latency_ms: float = 0.0
        self.fps_measured: float = 0.0
        
        # For sliding window FPS estimation
        self._timestamps: List[float] = []
        self._fps_window_size = 30

    def start(self) -> None:
        """
        Connects to the video capture device and launches the background loop.
        """
        with self.lock:
            if self.running:
                logger.warning(f"Camera stream {self.camera_idx} is already running.")
                return

            self._connect()
            self.running = True
            self.thread = threading.Thread(
                target=self._capture_loop, 
                name=f"CameraCapture-{self.camera_idx}", 
                daemon=True
            )
            self.thread.start()
            logger.info(f"Background capture thread started for camera {self.camera_idx}.")

    def _connect(self) -> None:
        """
        Attempts to open the VideoCapture device and configure resolution/FPS.
        """
        try:
            logger.info(f"Opening VideoCapture for index {self.camera_idx}...")
            self.cap = cv2.VideoCapture(self.camera_idx)
            
            # Configure hardware properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            if not self.cap.isOpened():
                self.is_connected = False
                logger.error(f"Failed to open camera device at index {self.camera_idx}")
            else:
                self.is_connected = True
                # Query actual parameters from hardware
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info(
                    f"Successfully connected camera {self.camera_idx}. "
                    f"Resolution set to {self.width}x{self.height} at target {self.target_fps} FPS."
                )
        except Exception as e:
            self.is_connected = False
            logger.exception(f"Unhandled exception during camera initialization: {e}")

    def _reconnect(self) -> None:
        """
        Closes current cap and attempts reconnect loop with basic backoff.
        """
        self.reconnect_count += 1
        logger.warning(f"Connection dropped. Attempting reconnect #{self.reconnect_count} for camera {self.camera_idx}...")
        
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            
        retry_delay = 1.5
        while self.running:
            self._connect()
            if self.is_connected:
                logger.info(f"Camera {self.camera_idx} reconnected successfully.")
                break
            logger.debug(f"Reconnect failed. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    def _capture_loop(self) -> None:
        """
        Dedicated acquisition thread loop.
        """
        delay = 1.0 / self.target_fps
        
        while True:
            with self.lock:
                if not self.running:
                    break

            if not self.is_connected or self.cap is None:
                self._reconnect()
                continue

            # Performance timing: measure read latency
            start_read = time.perf_counter()
            try:
                ret, frame = self.cap.read()
            except Exception as e:
                logger.error(f"Hardware read error: {e}")
                ret, frame = False, None
                
            latency_ms = (time.perf_counter() - start_read) * 1000.0
            
            now = time.time()

            if not ret or frame is None:
                with self.lock:
                    self.dropped_frame_count += 1
                logger.warning(f"Empty frame or read failure on camera {self.camera_idx}.")
                
                # If frame drops persist for over 2.0s, trigger reconnect
                if now - self.last_timestamp > 2.0 and self.last_timestamp > 0:
                    logger.error("No valid frames received for over 2 seconds. Triggering reconnection.")
                    self._reconnect()
                continue

            # Update thread-safe buffer and performance statistics
            with self.lock:
                self.frame = frame
                self.last_timestamp = now
                self.frame_count += 1
                self.read_latency_ms = latency_ms
                
                # sliding window FPS tracker
                self._timestamps.append(now)
                if len(self._timestamps) > self._fps_window_size:
                    self._timestamps.pop(0)
                    
                if len(self._timestamps) > 1:
                    duration = self._timestamps[-1] - self._timestamps[0]
                    if duration > 0:
                        self.fps_measured = (len(self._timestamps) - 1) / duration

            # Enforce target FPS pacing
            elapsed_read = time.perf_counter() - start_read
            sleep_time = max(0.001, delay - elapsed_read)
            time.sleep(sleep_time)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Fetches the latest frame, status, and acquisition timestamp.
        
        Returns:
            Tuple[bool, Optional[np.ndarray], float]: (is_active, frame, epoch_timestamp)
        """
        with self.lock:
            if not self.running or not self.is_connected or self.frame is None:
                return False, None, 0.0
            return True, self.frame.copy(), self.last_timestamp

    def get_health_status(self) -> Dict[str, Any]:
        """
        Gathers diagnostic performance metrics.
        """
        with self.lock:
            return {
                "camera_idx": self.camera_idx,
                "is_connected": self.is_connected,
                "measured_fps": round(self.fps_measured, 2),
                "read_latency_ms": round(self.read_latency_ms, 2),
                "total_frames_acquired": self.frame_count,
                "dropped_frames": self.dropped_frame_count,
                "reconnects_attempted": self.reconnect_count,
                "resolution": f"{self.width}x{self.height}" if self.is_connected else "0x0"
            }

    def stop(self) -> None:
        """
        Gracefully terminates the background thread and releases hardware connections.
        """
        logger.info(f"Stopping capture stream for camera {self.camera_idx}...")
        with self.lock:
            self.running = False
            
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        with self.lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.frame = None
            self.is_connected = False
            logger.info(f"Released capture resources for camera {self.camera_idx}.")


class CameraManager:
    """
    Orchestrates the lifecycle, selection, and health diagnostics of 
    multiple webcam stream devices.
    """
    def __init__(self, default_width: int = 640, default_height: int = 480, default_fps: int = 30):
        self.default_width = default_width
        self.default_height = default_height
        self.default_fps = default_fps
        self._streams: Dict[int, CameraStream] = {}
        self._lock = threading.Lock()

    def detect_available_cameras(self, max_check: int = 5) -> List[int]:
        """
        Queries system hardware to identify active webcam indices.
        
        Args:
            max_check: Total index boundaries to scan.
            
        Returns:
            List[int]: Active system webcam device indices.
        """
        active_indices = []
        logger.info(f"Scanning hardware indices 0 to {max_check - 1} for webcams...")
        
        for idx in range(max_check):
            cap = None
            try:
                # Open with headless fast-fail parameters
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        active_indices.append(idx)
                        logger.info(f"Detected functional camera index: {idx}")
            except Exception as e:
                logger.debug(f"Camera index {idx} not available: {e}")
            finally:
                if cap is not None:
                    cap.release()
                    
        return active_indices

    def get_stream(
        self, 
        camera_idx: int, 
        width: Optional[int] = None, 
        height: Optional[int] = None, 
        fps: Optional[int] = None
    ) -> CameraStream:
        """
        Initializes and returns a managed CameraStream instance.
        
        Args:
            camera_idx: Targeted camera index.
            width: Desired resolution width.
            height: Desired resolution height.
            fps: Desired capture frame pacing.
            
        Returns:
            CameraStream: Running capture stream instance.
        """
        w = width or self.default_width
        h = height or self.default_height
        f = fps or self.default_fps
        
        with self._lock:
            if camera_idx in self._streams:
                stream = self._streams[camera_idx]
                # If stream is running, check if parameters changed
                if stream.width != w or stream.height != h or stream.target_fps != f:
                    logger.info(f"Re-configuring active camera {camera_idx} to new configuration parameters.")
                    stream.stop()
                    stream.width = w
                    stream.height = h
                    stream.target_fps = f
                    stream.start()
                return stream
                
            logger.info(f"Registering new CameraStream for device {camera_idx}.")
            stream = CameraStream(camera_idx=camera_idx, width=w, height=h, target_fps=f)
            self._streams[camera_idx] = stream
            stream.start()
            return stream

    def release_stream(self, camera_idx: int) -> None:
        """
        Stops and unregisters a managed stream.
        """
        with self._lock:
            if camera_idx in self._streams:
                self._streams[camera_idx].stop()
                del self._streams[camera_idx]
                logger.info(f"Successfully released and removed camera stream {camera_idx}.")

    def shutdown(self) -> None:
        """
        Stops and releases all managed camera streams on application shutdown.
        """
        logger.info("Shutting down all active camera managers...")
        with self._lock:
            for idx, stream in list(self._streams.items()):
                stream.stop()
            self._streams.clear()
            logger.info("Successfully released all camera manager instances.")
