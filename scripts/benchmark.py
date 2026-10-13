"""Measure ONNX latency/FPS + model size. Usage: python scripts/benchmark.py [--model models/yolov8n.onnx] [--image x.jpg]"""
import argparse
import os
import sys
import time
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.detector import Detector

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/yolov8n.onnx")
p.add_argument("--image", default=None)
p.add_argument("--runs", type=int, default=100)
a = p.parse_args()

img = cv2.imread(a.image) if a.image else np.random.randint(0, 255, (720, 1280, 3), np.uint8)
d = Detector(a.model)
for _ in range(10):
    d.detect(img)  # warmup
t, inf = [], []
for _ in range(a.runs):
    s = time.perf_counter()
    _, ms = d.detect(img)
    t.append((time.perf_counter() - s) * 1000)
    inf.append(ms)
print(f"model size      : {os.path.getsize(a.model) / 1e6:.1f} MB")
print(f"provider        : {d.provider}")
print(f"inference only  : {np.mean(inf):.1f} ms (p95 {np.percentile(inf, 95):.1f})")
print(f"end-to-end      : {np.mean(t):.1f} ms  ->  {1000 / np.mean(t):.1f} FPS")
