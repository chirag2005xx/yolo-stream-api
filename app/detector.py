"""YOLOv8 ONNX inference: letterbox -> ORT -> decode -> NMS. No torch needed at runtime."""
import time
import cv2
import numpy as np
import onnxruntime as ort
from .classes import COCO


class Detector:
    def __init__(self, model_path: str, conf: float = 0.35, iou: float = 0.45):
        avail = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 640
        self.conf, self.iou = conf, iou
        self.provider = self.sess.get_providers()[0]

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
        return canvas, r, left, top

    def detect(self, bgr):
        """Returns (detections, inference_ms). Each detection: class_id, label, confidence, box [x1,y1,x2,y2]."""
        H, W = bgr.shape[:2]
        canvas, r, left, top = self._letterbox(bgr)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size), swapRB=True)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: blob})[0]
        ms = (time.perf_counter() - t0) * 1000

        preds = out[0].T  # (num_anchors, 4 + num_classes)
        scores = preds[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        preds, cls, conf = preds[keep], cls[keep], conf[keep]
        if len(preds) == 0:
            return [], ms

        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = np.clip((cx - w / 2 - left) / r, 0, W)
        y1 = np.clip((cy - h / 2 - top) / r, 0, H)
        x2 = np.clip((cx + w / 2 - left) / r, 0, W)
        y2 = np.clip((cy + h / 2 - top) / r, 0, H)
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
        idx = np.array(cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf, self.iou)).flatten()

        dets = []
        for i in idx:
            c = int(cls[i])
            dets.append({
                "class_id": c,
                "label": COCO[c] if c < len(COCO) else str(c),
                "confidence": round(float(conf[i]), 3),
                "box": [round(float(v), 1) for v in (x1[i], y1[i], x2[i], y2[i])],
            })
        return dets, ms


def annotate(frame, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        txt = f'{d["label"]} {d["confidence"]:.2f}'
        cv2.putText(frame, txt, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
