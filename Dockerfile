# ---- Stage 1: export YOLO weights to ONNX (torch lives only here) ----
FROM python:3.11-slim AS export
ARG MODEL=yolov8n
WORKDIR /w
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir ultralytics onnx
RUN python -c "from ultralytics import YOLO; YOLO('${MODEL}.pt').export(format='onnx', imgsz=640, opset=12)"

# ---- Stage 2: slim runtime image (onnxruntime + opencv only) ----
FROM python:3.11-slim
ARG MODEL=yolov8n
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY --from=export /w/${MODEL}.onnx /app/models/model.onnx
ENV MODEL_PATH=/app/models/model.onnx SOURCE=/data/sample.mp4
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
