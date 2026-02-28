# RunPod Backend (GPU)

Notebook-aligned backend for Perception Playground.

Endpoints:

- `POST /infer`
- `GET /health`

Supported tasks (and only these tasks):

- `object-detection` (Ultralytics YOLO detection)
- `image-segmentation` (Ultralytics YOLO instance segmentation)
- `pose-estimation` (Ultralytics YOLO human keypoints, 17 COCO joints)
- `depth-estimation` (Depth Anything small)
- `velocity-estimation` (RAFT optical flow + YOLO tracking)

## Request contract

```json
{
  "task": "object-detection",
  "model": "yolo26n.pt",
  "payloadBase64": "<base64 input>",
  "mimeType": "image/jpeg",
  "options": {
    "threshold": 0.3
  }
}
```

For `velocity-estimation`, upload video and optional options:

- `threshold` (YOLO tracking confidence, default `0.5`)
- `max_frames` (default `121`)
- `max_pairs` (default `120`)
- `meter_per_pixel` (default `0.05`)
- `imgsz` (default `640`)

## Run locally

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run with Docker (GPU host)

```bash
cd backend
docker build -t perception-backend:latest .
docker run --gpus all -p 8000:8000 perception-backend:latest
```

## Connect with frontend proxy

Set:

- `INFERENCE_BACKEND_URL=https://<your-host>/infer`

The Vercel API route forwards all supported playground requests to this backend.
