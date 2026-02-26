# RunPod Backend (GPU)

This backend provides a single endpoint used by the Vercel frontend proxy:

- `POST /infer`
- `GET /health`

It supports:

- `image-classification`
- `object-detection`
- `image-segmentation`
- `depth-estimation`
- `pose-estimation`
- `video-action-recognition`
- `sam-segmentation`
- `velocity-estimation`
- `perception-pipeline`

## Request contract

```json
{
  "task": "image-classification",
  "model": "google/vit-base-patch16-224",
  "payloadBase64": "<base64 input>",
  "mimeType": "image/jpeg",
  "options": {}
}
```

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
docker run --gpus all -p 8000:8000 -e HUGGINGFACE_API_KEY=hf_xxx perception-backend:latest
```

## RunPod without Docker (normal Pod)

If your RunPod setup does not use Docker images, run directly with Python:

```bash
git clone <your-repo-url>
cd <your-repo>/backend
chmod +x runpod_start.sh
export HUGGINGFACE_API_KEY=hf_xxx
./runpod_start.sh
```

Or run manually:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
export HUGGINGFACE_API_KEY=hf_xxx
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then expose port `8000` in RunPod and use:

`https://<runpod-id>-8000.proxy.runpod.net/infer`

## Deploy on RunPod

1. Create a GPU Pod (CUDA 12+ runtime).
2. Either:
   - deploy this backend container image, or
   - run backend directly using `runpod_start.sh` on the pod.
3. Expose port `8000` publicly.
4. Set env var:
   - `HUGGINGFACE_API_KEY` (used for pose fallback, optional but recommended).
5. Copy endpoint URL, e.g.:
   - `https://<runpod-id>-8000.proxy.runpod.net/infer`

## GPU and CUDA recommendation

- CUDA version: **12.1** (matches Docker image `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`)
- Minimum GPU for basic demos: **16 GB VRAM** (e.g., T4/A4000)
- Recommended for chained pipeline + smoother latency: **24 GB VRAM** (A10/L4/RTX 4090/L40S)
- For heavier concurrent usage/classroom demos: **40 GB+ VRAM** (A100 40GB)

## Connect with Vercel frontend

Set on Vercel:

- `INFERENCE_BACKEND_URL=https://<runpod-id>-8000.proxy.runpod.net/infer`

Then redeploy Vercel.

## Build-and-run quick path

```bash
cd backend
docker build -t perception-backend:latest .
docker run --gpus all -p 8000:8000 -e HUGGINGFACE_API_KEY=hf_xxx perception-backend:latest
curl http://localhost:8000/health
```
