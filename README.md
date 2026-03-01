# Perception Concept Studio

Perception Concept Studio is a React + Vercel frontend with a GPU backend for notebook-aligned and promptable computer vision inference.

## Implemented Playground Tasks

The backend and frontend are aligned to the validated `project_sdk/perception_project` notebooks, with pose estimation additionally enabled using Ultralytics YOLO pose models:

1. `01_kitti_tracking_detection_ultralytics.ipynb` -> `object-detection`
2. `02_kitti_instance_segmentation_ultralytics.ipynb` -> `image-segmentation`
3. `03_monocular_depth_ai.ipynb` -> `depth-estimation`
4. `04_motion_velocity_estimation.ipynb` -> `velocity-estimation`
5. YOLO Pose (`yolo26n-pose.pt`) -> `pose-estimation`
6. SAM 2 (`sam2.1_b.pt`) -> `sam2-segmentation` (image segmentation)

Any playground not backed by these implementations has been removed from the UI.

## Architecture

- Frontend: Vite + React (`src/`)
- API proxy: Vercel function (`api/hf-inference.ts`)
- GPU backend: FastAPI (`backend/`)

Request flow:

1. Frontend uploads image/video.
2. Vercel function forwards request to `INFERENCE_BACKEND_URL`.
3. FastAPI backend runs inference and returns JSON/video payload.

## Local Development

Frontend:

```sh
npm i
npm run dev
```

Backend:

```sh
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Set frontend env:

- `INFERENCE_BACKEND_URL=http://localhost:8000/infer`

## Notes

- Velocity estimates use Ultralytics `solutions.SpeedEstimator` with YOLO tracking and `meter_per_pixel` scaling.
- Default model fallbacks mirror notebook behavior (`yolo26*` -> `yolo11*` fallback when unavailable).
- Pose estimation uses the default 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles).
- SAM 2 models auto-download on first use (for example `sam2.1_b.pt`) with a compatible Ultralytics install.
- SAM 3-specific CLIP/tokenizer setup is no longer required for this app.


