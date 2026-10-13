"""Export YOLOv8 weights to ONNX. Usage: python scripts/export_onnx.py [--model yolov8n] [--imgsz 640]"""
import argparse
import os
import shutil
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--model", default="yolov8n")
p.add_argument("--imgsz", type=int, default=640)
a = p.parse_args()

out = YOLO(f"{a.model}.pt").export(format="onnx", imgsz=a.imgsz, opset=12, dynamic=False)
os.makedirs("models", exist_ok=True)
dest = os.path.join("models", f"{a.model}.onnx")
shutil.move(out, dest)
print(f"Saved {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
