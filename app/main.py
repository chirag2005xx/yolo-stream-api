import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from .detector import Detector, annotate
from .stream import StreamRunner

logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.onnx")
SOURCE = os.getenv("SOURCE", "0")  # 0 = webcam, rtsp://..., /path/video.mp4, or "none"
CONF = float(os.getenv("CONF", "0.35"))
IOU = float(os.getenv("IOU", "0.45"))

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run: python scripts/export_onnx.py")
    state["det"] = Detector(MODEL_PATH, CONF, IOU)
    state["runner"] = None
    if SOURCE.lower() != "none":
        state["runner"] = StreamRunner(SOURCE, state["det"])
        state["runner"].start()
    yield
    if state["runner"]:
        state["runner"].stop()


app = FastAPI(title="YOLO Stream API", version="1.0", lifespan=lifespan)

PAGE = """<!doctype html><title>YOLO Stream</title>
<body style="font-family:sans-serif;background:#111;color:#eee;text-align:center">
<h2>YOLOv8 + ONNX Runtime live stream</h2>
<img src="/video" style="max-width:90%;border:1px solid #444"><pre id="s"></pre>
<p><a style="color:#8cf" href="/docs">API docs</a></p>
<script>setInterval(async()=>{const r=await (await fetch('/metrics')).json();
document.getElementById('s').textContent=JSON.stringify(r,null,2)},1000)</script></body>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/health")
def health():
    r = state.get("runner")
    return {"status": "ok", "model": MODEL_PATH, "provider": state["det"].provider,
            "stream_connected": bool(r and r.connected)}


@app.get("/metrics")
def metrics():
    r = state.get("runner")
    if not r:
        raise HTTPException(404, "Stream disabled (SOURCE=none)")
    s = r.snapshot()
    s["detections"] = len(s["detections"])
    return s


@app.get("/detections")
def detections():
    r = state.get("runner")
    if not r:
        raise HTTPException(404, "Stream disabled (SOURCE=none)")
    return r.snapshot()


@app.get("/video")
def video():
    r = state.get("runner")
    if not r:
        raise HTTPException(404, "Stream disabled (SOURCE=none)")

    def gen():
        while True:
            jpg = r.latest_jpeg()
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.03)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/detect")
async def detect(file: UploadFile = File(...), annotated: bool = Query(False)):
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    dets, ms = state["det"].detect(img)
    if annotated:
        _, buf = cv2.imencode(".jpg", annotate(img, dets))
        return Response(buf.tobytes(), media_type="image/jpeg")
    return {"inference_ms": round(ms, 1), "count": len(dets), "detections": dets}
