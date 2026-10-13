"""Background thread: read webcam / RTSP / video file, run detection, keep latest annotated frame + stats."""
import logging
import os
import threading
import time
import cv2
from .detector import Detector, annotate

log = logging.getLogger("stream")
# Prefer TCP for RTSP: far fewer corrupted frames than UDP
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class StreamRunner(threading.Thread):
    def __init__(self, source, detector: Detector):
        super().__init__(daemon=True)
        self.source = int(source) if str(source).isdigit() else source
        self.det = detector
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self.jpeg = None
        self.detections = []
        self.fps = 0.0
        self.infer_ms = 0.0
        self.frames = 0
        self.connected = False

    def stop(self):
        self._stop_evt.set()

    def snapshot(self):
        with self._lock:
            return {
                "source": str(self.source), "connected": self.connected, "frames": self.frames,
                "fps": round(self.fps, 1), "inference_ms": round(self.infer_ms, 1),
                "detections": list(self.detections),
            }

    def latest_jpeg(self):
        with self._lock:
            return self.jpeg

    def run(self):
        while not self._stop_evt.is_set():
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                log.warning("Cannot open source %s, retrying in 3s", self.source)
                self.connected = False
                time.sleep(3)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.connected = True
            log.info("Connected to %s", self.source)
            last = time.perf_counter()
            while not self._stop_evt.is_set():
                ok, frame = cap.read()
                if not ok:
                    break  # end of file or dropped stream -> reopen (files loop, RTSP reconnects)
                dets, ms = self.det.detect(frame)
                ok_enc, buf = cv2.imencode(".jpg", annotate(frame, dets), [cv2.IMWRITE_JPEG_QUALITY, 80])
                now = time.perf_counter()
                inst = 1.0 / max(now - last, 1e-6)
                last = now
                with self._lock:
                    self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
                    self.infer_ms = ms
                    self.frames += 1
                    self.detections = dets
                    if ok_enc:
                        self.jpeg = buf.tobytes()
            cap.release()
            self.connected = False
            time.sleep(0.2 if isinstance(self.source, str) and os.path.isfile(self.source) else 1.0)
