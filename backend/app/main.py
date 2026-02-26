import base64
import io
import logging
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    DetrImageProcessor,
    DetrForObjectDetection,
    pipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception-backend")

DEFAULT_MODELS: Dict[str, str] = {
    "image-classification": "google/vit-base-patch16-224",
    "object-detection": "facebook/detr-resnet-50",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "depth-estimation": "Intel/dpt-large",
    "pose-estimation": "usyd-community/vitpose-base-simple",
    "video-action-recognition": "MCG-NJU/videomae-base-finetuned-kinetics",
    "sam-segmentation": "facebook/sam-vit-base",
    "velocity-estimation": "facebook/detr-resnet-50",
    "perception-pipeline": "facebook/detr-resnet-50",
}


class InferenceRequest(BaseModel):
    task: str
    payloadBase64: str = Field(..., min_length=8)
    mimeType: str = "application/octet-stream"
    model: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)


def get_device() -> int | str:
    return 0 if torch.cuda.is_available() else "cpu"


def decode_image(payload_base64: str) -> Image.Image:
    raw = base64.b64decode(payload_base64)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


def image_to_base64_png(image: Image.Image) -> str:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return base64.b64encode(out.getvalue()).decode("utf-8")


def decode_video_to_tempfile(payload_base64: str) -> str:
    raw = base64.b64decode(payload_base64)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(raw)
    tmp.flush()
    tmp.close()
    return tmp.name


def read_video_frames(video_path: str, max_frames: int = 60, stride: int = 1) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video input.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames: List[np.ndarray] = []
    idx = 0
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError("No frames decoded from video.")
    return frames, float(fps)


def resize_frame(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)))


def encode_video_base64(frames: List[np.ndarray], fps: float = 24.0) -> str:
    if not frames:
        raise RuntimeError("No frames to encode.")
    h, w = frames[0].shape[:2]
    out_tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    out_path = out_tmp.name
    out_tmp.close()

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Failed to initialize video writer (mp4v/XVID).")
    for frame in frames:
        writer.write(frame)
    writer.release()

    with open(out_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    os.remove(out_path)
    return encoded


@lru_cache(maxsize=16)
def get_classifier(model_id: str):
    return pipeline("image-classification", model=model_id, device=get_device())


@lru_cache(maxsize=16)
def get_detector(model_id: str):
    processor = DetrImageProcessor.from_pretrained(model_id)
    model = DetrForObjectDetection.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


@lru_cache(maxsize=16)
def get_segmentation(model_id: str):
    return pipeline("image-segmentation", model=model_id, device=get_device())


@lru_cache(maxsize=16)
def get_depth(model_id: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


@lru_cache(maxsize=16)
def get_video_classifier(model_id: str):
    return pipeline("video-classification", model=model_id, device=get_device())


@lru_cache(maxsize=16)
def get_sam(model_id: str):
    return pipeline("mask-generation", model=model_id, device=get_device())


def get_tracker():
    return DeepSort(max_age=30, n_init=2, nms_max_overlap=0.8)


@lru_cache(maxsize=8)
def get_vitpose(model_id: str):
    # Some transformer releases may not have VitPose classes.
    # Load lazily and fail with a clean error so caller can fallback to HF API.
    from transformers import VitPoseForPoseEstimation  # type: ignore

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = VitPoseForPoseEstimation.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return processor, model


def normalize_box(box: Dict[str, Any]) -> Dict[str, float]:
    return {
        "xmin": float(box.get("xmin", 0)),
        "ymin": float(box.get("ymin", 0)),
        "xmax": float(box.get("xmax", 0)),
        "ymax": float(box.get("ymax", 0)),
    }


def run_image_classification(image: Image.Image, model_id: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_k = int(options.get("top_k", 5))
    classifier = get_classifier(model_id)
    result = classifier(image, top_k=top_k)
    return [{"label": item["label"], "score": float(item["score"])} for item in result]


def run_object_detection(image: Image.Image, model_id: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    threshold = float(options.get("threshold", 0.25))
    processor, model = get_detector(model_id)
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=outputs.logits.device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detections: List[Dict[str, Any]] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detections.append(
            {
                "score": float(score.item()),
                "label": model.config.id2label[int(label.item())],
                "box": {
                    "xmin": float(box[0].item()),
                    "ymin": float(box[1].item()),
                    "xmax": float(box[2].item()),
                    "ymax": float(box[3].item()),
                },
            }
        )
    return detections


def run_segmentation(image: Image.Image, model_id: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    threshold = float(options.get("threshold", 0.2))
    segmenter = get_segmentation(model_id)
    results = segmenter(image, threshold=threshold)
    compact: List[Dict[str, Any]] = []
    for item in results:
        compact.append(
            {
                "label": item.get("label", "segment"),
                "score": float(item.get("score", 0.0)),
            }
        )
    return compact


def run_depth_estimation(image: Image.Image, model_id: str) -> Dict[str, str]:
    processor, model = get_depth(model_id)
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    output = prediction.cpu().numpy()
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    depth_image = Image.fromarray((output * 255).astype(np.uint8))
    return {"depth_image": image_to_base64_png(depth_image)}


def run_video_action(payload_base64: str, model_id: str, options: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_k = int(options.get("top_k", 5))
    classifier = get_video_classifier(model_id)
    raw = base64.b64decode(payload_base64)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        results = classifier(tmp.name, top_k=top_k)
    return [{"label": item["label"], "score": float(item["score"])} for item in results]


def run_velocity_estimation(payload_base64: str, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    max_frames = int(options.get("max_frames", 48))
    stride = int(options.get("stride", 1))
    threshold = float(options.get("threshold", 0.35))

    in_path = decode_video_to_tempfile(payload_base64)
    frames, fps = read_video_frames(in_path, max_frames=max_frames, stride=stride)
    os.remove(in_path)

    tracker = get_tracker()
    last_centers: Dict[int, Tuple[float, float]] = {}
    velocity_map: Dict[int, float] = {}
    annotated: List[np.ndarray] = []

    for frame in frames:
        frame = resize_frame(frame, max_width=int(options.get("max_width", 960)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        detections = run_object_detection(pil, model_id, {"threshold": threshold})

        ds_inputs = []
        for det in detections:
            box = det["box"]
            x, y = box["xmin"], box["ymin"]
            w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
            ds_inputs.append(([x, y, w, h], float(det["score"]), det["label"]))

        tracks = tracker.update_tracks(ds_inputs, frame=frame)
        overlay = frame.copy()
        for trk in tracks:
            if not trk.is_confirmed():
                continue
            ltrb = trk.to_ltrb()
            track_id = int(trk.track_id)
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            prev = last_centers.get(track_id)
            if prev is not None:
                dist_px = float(np.hypot(cx - prev[0], cy - prev[1]))
                velocity_map[track_id] = dist_px * fps
            last_centers[track_id] = (cx, cy)
            vel = velocity_map.get(track_id, 0.0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 240, 255), 2)
            cv2.putText(
                overlay,
                f"ID {track_id} | {vel:.1f} px/s",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 240, 255),
                2,
                cv2.LINE_AA,
            )
        annotated.append(overlay)

    avg_vel = float(np.mean(list(velocity_map.values()))) if velocity_map else 0.0
    return {
        "annotated_video": encode_video_base64(annotated, fps=fps),
        "content_type": "video/mp4",
        "metrics": {
            "frames_processed": len(frames),
            "tracks_count": len(velocity_map),
            "avg_velocity_px_s": avg_vel,
        },
    }


def run_perception_pipeline(payload_base64: str, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    max_frames = int(options.get("max_frames", 28))
    stride = int(options.get("stride", 1))
    threshold = float(options.get("threshold", 0.35))
    depth_model_id = options.get("depth_model", DEFAULT_MODELS["depth-estimation"])
    seg_model_id = options.get("seg_model", DEFAULT_MODELS["image-segmentation"])

    in_path = decode_video_to_tempfile(payload_base64)
    frames, fps = read_video_frames(in_path, max_frames=max_frames, stride=stride)
    os.remove(in_path)

    tracker = get_tracker()
    last_centers: Dict[int, Tuple[float, float]] = {}
    velocity_map: Dict[int, float] = {}
    depth_values: List[float] = []
    annotated: List[np.ndarray] = []

    for idx, frame in enumerate(frames):
        frame = resize_frame(frame, max_width=int(options.get("max_width", 960)))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        detections = run_object_detection(pil, model_id, {"threshold": threshold})
        segments = run_segmentation(pil, seg_model_id, {"threshold": 0.3})
        depth = run_depth_estimation(pil, depth_model_id)

        depth_img = decode_image(depth["depth_image"])
        depth_arr = np.array(depth_img).astype(np.float32)
        depth_values.append(float(depth_arr.mean()))

        ds_inputs = []
        for det in detections:
            box = det["box"]
            x, y = box["xmin"], box["ymin"]
            w, h = box["xmax"] - box["xmin"], box["ymax"] - box["ymin"]
            ds_inputs.append(([x, y, w, h], float(det["score"]), det["label"]))
        tracks = tracker.update_tracks(ds_inputs, frame=frame)

        overlay = frame.copy()
        cv2.putText(
            overlay,
            f"Frame {idx + 1} | Segments {len(segments)} | Depth(mean) {depth_values[-1]:.1f}",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (40, 255, 40),
            2,
            cv2.LINE_AA,
        )

        for trk in tracks:
            if not trk.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, trk.to_ltrb())
            track_id = int(trk.track_id)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            prev = last_centers.get(track_id)
            if prev is not None:
                dist_px = float(np.hypot(cx - prev[0], cy - prev[1]))
                velocity_map[track_id] = dist_px * fps
            last_centers[track_id] = (cx, cy)
            vel = velocity_map.get(track_id, 0.0)

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 140, 0), 2)
            cv2.putText(
                overlay,
                f"ID {track_id} {vel:.1f}px/s",
                (x1, max(12, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 140, 0),
                2,
                cv2.LINE_AA,
            )

        # Show depth thumbnail for intuitive visualization.
        depth_bgr = cv2.cvtColor(np.array(depth_img), cv2.COLOR_GRAY2BGR)
        depth_bgr = cv2.resize(depth_bgr, (160, 90))
        overlay[10:100, overlay.shape[1] - 170:overlay.shape[1] - 10] = depth_bgr
        cv2.rectangle(overlay, (overlay.shape[1] - 170, 10), (overlay.shape[1] - 10, 100), (255, 255, 255), 1)
        cv2.putText(
            overlay,
            "Depth",
            (overlay.shape[1] - 165, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        annotated.append(overlay)

    return {
        "annotated_video": encode_video_base64(annotated, fps=fps),
        "content_type": "video/mp4",
        "metrics": {
            "frames_processed": len(frames),
            "tracks_count": len(velocity_map),
            "avg_velocity_px_s": float(np.mean(list(velocity_map.values()))) if velocity_map else 0.0,
            "avg_depth_intensity": float(np.mean(depth_values)) if depth_values else 0.0,
        },
    }


def run_sam(image: Image.Image, model_id: str) -> List[Dict[str, Any]]:
    generator = get_sam(model_id)
    results = generator(image)
    masks = results if isinstance(results, list) else results.get("masks", [])
    compact: List[Dict[str, Any]] = []
    for item in masks[:10]:
        box = item.get("bbox", {})
        compact.append(
            {
                "label": item.get("label", "mask"),
                "score": float(item.get("score", 0.0)),
                "box": normalize_box(box) if isinstance(box, dict) else None,
            }
        )
    return compact


def run_pose_local(image: Image.Image, model_id: str) -> List[Dict[str, Any]]:
    processor, model = get_vitpose(model_id)
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    if not hasattr(processor, "post_process_pose_estimation"):
        raise RuntimeError("Pose post-processing is unavailable in this transformers version.")
    pose = processor.post_process_pose_estimation(outputs, target_sizes=[image.size[::-1]])[0]
    people: List[Dict[str, Any]] = []
    for person in pose:
        people.append(
            {
                "score": float(person.get("score", 0.0)),
                "keypoints": person.get("keypoints", []),
            }
        )
    return people


def run_pose_hf_fallback(payload_base64: str, mime_type: str, model_id: str) -> List[Dict[str, Any]]:
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise RuntimeError("Pose local load failed and HUGGINGFACE_API_KEY not set for fallback.")
    binary = base64.b64decode(payload_base64)
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers={"Authorization": f"Bearer {hf_token}", "Content-Type": mime_type or "image/jpeg"},
        data=binary,
        timeout=120,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"HF pose fallback failed ({response.status_code}): {response.text[:300]}")
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError("HF pose fallback returned unexpected response type.")
    return payload


app = FastAPI(title="Perception Playground Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True, "cuda": torch.cuda.is_available()}


@app.post("/infer")
def infer(request: InferenceRequest):
    task = request.task
    model_id = request.model or DEFAULT_MODELS.get(task)
    if not model_id:
        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")

    try:
        if task == "video-action-recognition":
            return run_video_action(request.payloadBase64, model_id, request.options)
        if task == "velocity-estimation":
            return run_velocity_estimation(request.payloadBase64, model_id, request.options)
        if task == "perception-pipeline":
            return run_perception_pipeline(request.payloadBase64, model_id, request.options)

        image = decode_image(request.payloadBase64)

        if task == "image-classification":
            return run_image_classification(image, model_id, request.options)
        if task == "object-detection":
            return run_object_detection(image, model_id, request.options)
        if task == "image-segmentation":
            return run_segmentation(image, model_id, request.options)
        if task == "depth-estimation":
            return run_depth_estimation(image, model_id)
        if task == "sam-segmentation":
            return run_sam(image, model_id)
        if task == "pose-estimation":
            try:
                return run_pose_local(image, model_id)
            except Exception:
                return run_pose_hf_fallback(request.payloadBase64, request.mimeType, model_id)

        raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed for task=%s model=%s", task, model_id)
        raise HTTPException(status_code=500, detail=str(exc))
