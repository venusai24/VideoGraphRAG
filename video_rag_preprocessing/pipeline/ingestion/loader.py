import numpy as np
import time

class VideoIngestor:
    """Mock video ingestor mimicking PyAV/OpenCV frame streaming."""
    def __init__(self, video_path: str, native_fps: float = 30.0, duration: float = 5.0):
        self.video_path = video_path
        self.native_fps = native_fps
        self.duration = duration

    def stream_frames(self):
        """Yields mock frames and monotonically increasing timestamps."""
        total_frames = int(self.duration * self.native_fps)
        for i in range(total_frames):
            timestamp = i / self.native_fps
            # Simulated blank image payload
            frame_data = np.zeros((224, 224, 3), dtype=np.uint8)
            yield frame_data, timestamp
