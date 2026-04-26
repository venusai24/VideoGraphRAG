
import cv2
import time

class VideoIngestor:
    """Video ingestor using OpenCV frame streaming."""
    def __init__(self, video_path: str, native_fps: float = None, duration: float = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.native_fps = fps if fps > 0 and native_fps is None else (native_fps or 30.0)
        self.duration = duration

    def stream_frames(self):
        """Yields real frames and monotonically increasing timestamps."""
        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / self.native_fps
            if self.duration is not None and timestamp > self.duration:
                break
                
            yield frame, timestamp
            frame_idx += 1
            
        self.cap.release()
