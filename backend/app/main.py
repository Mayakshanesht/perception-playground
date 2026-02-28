import base64
import io
import logging
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import pipeline
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception-backend")

DEFAULT_MODELS: Dict[str, str] = {
    "object-detection": "yolo26n.pt",
    "image-segmentation": "yolo26n-seg.pt",
    "pose-estimation": "yolo26n-pose.pt",
    "sam3-concept-segmentation": "sam3.pt",
    "depth-estimation": "LiheYoung/depth-anything-small-hf",
    "velocity-estimation": "raft-large",
}


class InferenceRequest(BaseModel):
    task: str
    payloadBase64: str = Field(..., min_length=8)
    mimeType: str = "application/octet-stream"
    model: str | None = None
    options: Dict[str, Any] = Field(default_factory=dict)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_image(payload_base64: str) -> Image.Image:
    raw = base64.b64decode(payload_base64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def decode_video_to_tempfile(payload_base64: str) -> str:
    raw = base64.b64decode(payload_base64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name


def image_to_base64_png(image: Image.Image) -> str:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("utf-8")


def encode_video_base64(frames: List[np.ndarray], fps: float) -> str:
    if not frames:
        raise RuntimeError("No frames available for video encoding.")

    h, w = frames[0].shape[:2]
    w = w if w % 2 == 0 else w - 1
    h = h if h % 2 == 0 else h - 1

    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tmp.name
    out_tmp.close()

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to open mp4 encoder.")

    for frame in frames:
        writer.write(np.ascontiguousarray(frame[:h, :w].astype(np.uint8)))
    writer.release()

    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    os.remove(out_path)
    return encoded


def safe_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


def safe_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


@lru_cache(maxsize=2)
def get_detection_model() -> YOLO:
    try:
        logger.info("Loading YOLO detection model yolo26n.pt")
        return YOLO("yolo26n.pt")
    except Exception:
        logger.warning("Could not load yolo26n.pt, falling back to yolo11n.pt")
        return YOLO("yolo11n.pt")


@lru_cache(maxsize=2)
def get_segmentation_model() -> YOLO:
    try:
        logger.info("Loading YOLO segmentation model yolo26n-seg.pt")
        return YOLO("yolo26n-seg.pt")
    except Exception:
        logger.warning("Could not load yolo26n-seg.pt, falling back to yolo11n-seg.pt")
        return YOLO("yolo11n-seg.pt")


@lru_cache(maxsize=2)
def get_pose_model() -> YOLO:
    try:
        logger.info("Loading YOLO pose model yolo26n-pose.pt")
        return YOLO("yolo26n-pose.pt")
    except Exception:
        logger.warning("Could not load yolo26n-pose.pt, falling back to yolo11n-pose.pt")
        return YOLO("yolo11n-pose.pt")


@lru_cache(maxsize=4)
def get_depth_pipeline(model_id: str):
    return pipeline(
        "depth-estimation",
        model=model_id,
        device=0 if torch.cuda.is_available() else -1,
    )


@lru_cache(maxsize=1)
def get_raft_bundle():
    device = get_device()
    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_large(weights=weights, progress=False).to(device).eval()
    return model, transforms, device


def extract_instances_from_result(r: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if r is None or r.boxes is None:
        return output

    boxes = r.boxes.xyxy.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, "cls") else np.zeros((len(boxes),), dtype=int)
    scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") else np.ones((len(boxes),), dtype=float)
    masks = r.masks.data.cpu().numpy() if r.masks is not None else None
    mask_polys = r.masks.xy if r.masks is not None and hasattr(r.masks, "xy") else None
    names = getattr(r, "names", {})

    for i, b in enumerate(boxes):
        c = int(classes[i]) if i < len(classes) else 0
        s = float(scores[i]) if i < len(scores) else 1.0
        label = names.get(c, str(c)) if isinstance(names, dict) else str(c)
        item: Dict[str, Any] = {
            "instance_id": i,
            "bbox": [float(x) for x in b.tolist()],
            "class_id": c,
            "class_name": label,
            "confidence": s,
            "label": label,
            "score": s,
            "box": {
                "xmin": float(b[0]),
                "ymin": float(b[1]),
                "xmax": float(b[2]),
                "ymax": float(b[3]),
            },
        }
        if masks is not None and i < len(masks):
            item["mask_area_px"] = int((masks[i] > 0.5).sum())
        if mask_polys is not None and i < len(mask_polys):
            poly = mask_polys[i]
            if poly is not None and len(poly) > 2:
                item["mask_polygon"] = [[float(pt[0]), float(pt[1])] for pt in poly.tolist()]
        output.append(item)

    return output


def run_object_detection(image: Image.Image, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    conf = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    imgsz = safe_int(options.get("imgsz", 640), 640, 320, 1280)

    model = get_detection_model()
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(source=bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []

    return extract_instances_from_result(results[0])


def run_segmentation(image: Image.Image, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    conf = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    imgsz = safe_int(options.get("imgsz", 640), 640, 320, 1280)

    model = get_segmentation_model()
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(source=bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []

    return extract_instances_from_result(results[0])


def run_depth_estimation(image: Image.Image, model_id: str) -> Dict[str, str]:
    depth_pipe = get_depth_pipeline(model_id)
    pred = depth_pipe(image)
    depth = np.array(pred["depth"], dtype=np.float32)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_image = Image.fromarray((depth * 255).astype(np.uint8))
    return {"depth_image": image_to_base64_png(depth_image)}


def parse_text_prompts(options: Dict[str, Any]) -> List[str]:
    raw = str(options.get("text_prompt", "")).strip() or str(options.get("text", "")).strip()
    if not raw:
        raw = "person"
    prompts = [x.strip() for x in raw.split(",") if x.strip()]
    return prompts or ["person"]


def run_sam3_concept_image(image: Image.Image, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    threshold = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    prompts = parse_text_prompts(options)

    img_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img_path = img_tmp.name
    img_tmp.close()
    try:
        image.save(img_path, format="JPEG")
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor  # type: ignore
        except Exception as exc:
            raise RuntimeError("SAM3SemanticPredictor unavailable. Install ultralytics>=8.3.237.") from exc

        predictor = SAM3SemanticPredictor(
            overrides={
                "conf": threshold,
                "task": "segment",
                "mode": "predict",
                "model": model_id,
                "verbose": False,
            }
        )
        predictor.set_image(img_path)
        results = predictor(text=prompts)
        if not results:
            return {"concepts": prompts, "instances": []}
        return {"concepts": prompts, "instances": extract_instances_from_result(results[0])}
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def run_sam3_concept_video(payload_base64: str, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    threshold = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    prompts = parse_text_prompts(options)
    max_frames = safe_int(options.get("max_frames", 90), 90, 4, 300)

    in_path = decode_video_to_tempfile(payload_base64)
    try:
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor  # type: ignore
        except Exception as exc:
            raise RuntimeError("SAM3VideoSemanticPredictor unavailable. Install ultralytics>=8.3.237.") from exc

        predictor = SAM3VideoSemanticPredictor(
            overrides={
                "conf": threshold,
                "task": "segment",
                "mode": "predict",
                "model": model_id,
                "verbose": False,
            }
        )
        stream = predictor(source=in_path, text=prompts, stream=True)

        frames: List[np.ndarray] = []
        total_instances = 0
        for idx, r in enumerate(stream):
            frames.append(r.plot())
            total_instances += int(len(r.boxes)) if getattr(r, "boxes", None) is not None else 0
            if idx + 1 >= max_frames:
                break

        if not frames:
            raise RuntimeError("SAM3 video segmentation returned no frames.")

        return {
            "concepts": prompts,
            "annotated_video": encode_video_base64(frames, fps=10.0),
            "content_type": "video/mp4",
            "metrics": {
                "frames_processed": len(frames),
                "instances_detected_total": total_instances,
            },
        }
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)


def run_pose_estimation(image: Image.Image, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    conf = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    imgsz = safe_int(options.get("imgsz", 640), 640, 320, 1280)

    model = get_pose_model()
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    results = model.predict(source=bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []

    r = results[0]
    output: List[Dict[str, Any]] = []
    if r.keypoints is None or r.boxes is None:
        return output

    kp_data = r.keypoints.data.cpu().numpy()
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    classes = r.boxes.cls.cpu().numpy().astype(int)

    for i in range(min(len(kp_data), len(boxes), len(scores), len(classes))):
        kpts = []
        for kp in kp_data[i]:
            if len(kp) >= 3:
                kpts.append([float(kp[0]), float(kp[1]), float(kp[2])])
            else:
                kpts.append([float(kp[0]), float(kp[1]), 1.0])
        b = boxes[i]
        c = classes[i]
        output.append(
            {
                "score": float(scores[i]),
                "label": r.names[int(c)],
                "class_id": int(c),
                "box": {
                    "xmin": float(b[0]),
                    "ymin": float(b[1]),
                    "xmax": float(b[2]),
                    "ymax": float(b[3]),
                },
                "keypoints": kpts,
            }
        )

    return output


def flow_to_color(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * 8, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), mag


def read_video_frames(video_path: str, max_frames: int) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames: list[np.ndarray] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Video must contain at least two frames.")

    return frames, float(fps)


def run_yolo_tracking(video_path: str, conf: float, imgsz: int) -> tuple[list[np.ndarray], list[Dict[str, Any]]]:
    model = get_detection_model()
    try:
        stream = model.track(
            source=video_path,
            tracker="botsort.yaml",
            persist=True,
            conf=conf,
            iou=0.4,
            imgsz=imgsz,
            stream=True,
            verbose=False,
        )
    except Exception:
        logger.warning("BoT-SORT tracking failed, retrying with default tracker config")
        stream = model.track(
            source=video_path,
            persist=True,
            conf=conf,
            iou=0.4,
            imgsz=imgsz,
            stream=True,
            verbose=False,
        )

    annotated_frames: list[np.ndarray] = []
    all_tracks: list[Dict[str, Any]] = []

    for frame_id, r in enumerate(stream):
        annotated_frames.append(r.plot())
        frame_tracks: Dict[str, Any] = {"frame_id": frame_id, "tracks": []}

        if r.boxes is not None and r.boxes.id is not None:
            ids = r.boxes.id
            boxes = r.boxes.xyxy
            classes = r.boxes.cls
            scores = r.boxes.conf

            for i in range(len(ids)):
                cid = int(classes[i].item())
                frame_tracks["tracks"].append(
                    {
                        "track_id": int(ids[i].item()),
                        "bbox": boxes[i].cpu().numpy().tolist(),
                        "class_id": cid,
                        "class_name": r.names[cid],
                        "confidence": float(scores[i].item()),
                    }
                )

        all_tracks.append(frame_tracks)

    return annotated_frames, all_tracks


def run_raft_velocity(
    frames_bgr: list[np.ndarray],
    tracks: list[Dict[str, Any]],
    fps: float,
    meter_per_pixel: float,
    max_pairs: int,
) -> tuple[list[np.ndarray], Dict[str, Any]]:
    model, transforms, device = get_raft_bundle()

    pair_count = min(len(frames_bgr) - 1, max_pairs)
    flow_vis_frames: list[np.ndarray] = []
    velocity_data: list[Dict[str, Any]] = []
    global_mags: list[float] = []

    for i in range(pair_count):
        im1_bgr = frames_bgr[i]
        im2_bgr = frames_bgr[i + 1]
        im1 = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2RGB)
        im2 = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2RGB)

        t1 = torch.from_numpy(im1).permute(2, 0, 1).float()[None] / 255.0
        t2 = torch.from_numpy(im2).permute(2, 0, 1).float()[None] / 255.0
        t1, t2 = transforms(t1, t2)
        t1 = t1.to(device)
        t2 = t2.to(device)

        with torch.no_grad():
            flow_list = model(t1, t2)
        flow = flow_list[-1][0].permute(1, 2, 0).cpu().numpy()

        flow_bgr, mag = flow_to_color(flow)
        global_mean_flow = float(mag.mean())
        global_mags.append(global_mean_flow)

        rec: Dict[str, Any] = {"frame_id": i, "tracks": []}
        frame_tracks = tracks[i].get("tracks", []) if i < len(tracks) else []

        for tr in frame_tracks:
            x1, y1, x2, y2 = tr["bbox"]
            x1 = int(max(0, min(flow.shape[1] - 1, x1)))
            x2 = int(max(0, min(flow.shape[1] - 1, x2)))
            y1 = int(max(0, min(flow.shape[0] - 1, y1)))
            y2 = int(max(0, min(flow.shape[0] - 1, y2)))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = flow[y1:y2, x1:x2, :]
            mag_roi = np.sqrt((roi[..., 0] ** 2) + (roi[..., 1] ** 2))
            px_per_frame = float(np.median(mag_roi))
            speed_mps = float(px_per_frame * fps * meter_per_pixel)
            rec["tracks"].append(
                {
                    "track_id": tr.get("track_id", -1),
                    "class_name": tr.get("class_name", ""),
                    "median_flow_px_per_frame": px_per_frame,
                    "speed_mps_est": speed_mps,
                }
            )

        velocity_data.append(rec)

        panel = cv2.hconcat([im1_bgr, flow_bgr])
        cv2.putText(
            panel,
            f"pair {i:04d}  mean_flow_px={global_mean_flow:.3f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        flow_vis_frames.append(panel)

    mean_global = float(np.mean(global_mags)) if global_mags else 0.0
    metrics = {
        "frames_processed": len(flow_vis_frames),
        "mean_flow_px_per_frame": mean_global,
        "mean_speed_mps_est": float(mean_global * fps * meter_per_pixel),
    }

    return flow_vis_frames, {"mode": "per_track", "data": velocity_data, "metrics": metrics}


def run_velocity_estimation(payload_base64: str, options: Dict[str, Any]) -> Dict[str, Any]:
    max_frames = safe_int(options.get("max_frames", 121), 121, 4, 360)
    max_pairs = safe_int(options.get("max_pairs", 120), 120, 1, 300)
    conf = safe_float(options.get("threshold", 0.5), 0.5, 0.01, 0.95)
    meter_per_pixel = safe_float(options.get("meter_per_pixel", 0.05), 0.05, 0.0001, 10.0)
    imgsz = safe_int(options.get("imgsz", 640), 640, 320, 1280)

    in_path = decode_video_to_tempfile(payload_base64)
    try:
        frames_bgr, fps = read_video_frames(in_path, max_frames=max_frames)
        _, tracks = run_yolo_tracking(in_path, conf=conf, imgsz=imgsz)
        flow_vis_frames, velocity_info = run_raft_velocity(
            frames_bgr=frames_bgr,
            tracks=tracks,
            fps=fps,
            meter_per_pixel=meter_per_pixel,
            max_pairs=max_pairs,
        )
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)

    return {
        "annotated_video": encode_video_base64(flow_vis_frames, fps=fps),
        "content_type": "video/mp4",
        "velocity": {
            "fps": fps,
            "meter_per_pixel": meter_per_pixel,
            "mode": velocity_info["mode"],
            "data": velocity_info["data"],
        },
        "metrics": velocity_info["metrics"],
    }


app = FastAPI(title="Perception Playground Backend", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available(), "tasks": list(DEFAULT_MODELS.keys())}


@app.post("/infer")
def infer(request: InferenceRequest):
    task = request.task
    model_id = request.model or DEFAULT_MODELS.get(task)
    if task not in DEFAULT_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported task: {task}. Supported tasks: {', '.join(DEFAULT_MODELS.keys())}",
        )

    try:
        if task == "velocity-estimation":
            return run_velocity_estimation(request.payloadBase64, request.options)
        if task == "sam3-concept-segmentation":
            if (request.mimeType or "").lower().startswith("video/"):
                return run_sam3_concept_video(request.payloadBase64, model_id, request.options)
            image = decode_image(request.payloadBase64)
            return run_sam3_concept_image(image, model_id, request.options)

        image = decode_image(request.payloadBase64)

        if task == "object-detection":
            return run_object_detection(image, request.options)
        if task == "image-segmentation":
            return run_segmentation(image, request.options)
        if task == "pose-estimation":
            return run_pose_estimation(image, request.options)
        if task == "depth-estimation":
            return run_depth_estimation(image, model_id)

        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed for task=%s model=%s", task, model_id)
        raise HTTPException(status_code=500, detail=str(exc))
