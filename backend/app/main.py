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
from transformers import pipeline
from ultralytics import SAM, YOLO, solutions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception-backend")

DEFAULT_MODELS: Dict[str, str] = {
    "object-detection": "yolo26n.pt",
    "image-segmentation": "yolo26n-seg.pt",
    "pose-estimation": "yolo26n-pose.pt",
    "sam2-segmentation": "sam2.1_b.pt",
    "depth-estimation": "LiheYoung/depth-anything-small-hf",
    "velocity-estimation": "yolo26n.pt",
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

    writer = None
    for codec in ("avc1", "H264", "mp4v"):
        candidate = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError("Failed to open video encoder (avc1/H264/mp4v).")

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


def safe_list_of_ints(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out or None


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
def get_sam2_model(model_id: str) -> SAM:
    logger.info("Loading SAM2 model %s", model_id)
    return SAM(model_id)


@lru_cache(maxsize=4)
def get_depth_pipeline(model_id: str):
    return pipeline(
        "depth-estimation",
        model=model_id,
        device=0 if torch.cuda.is_available() else -1,
    )


def extract_instances_from_result(r: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    if r is None:
        return output

    names = getattr(r, "names", {}) if hasattr(r, "names") else {}

    boxes = r.boxes.xyxy.cpu().numpy() if getattr(r, "boxes", None) is not None else None
    classes = r.boxes.cls.cpu().numpy().astype(int) if getattr(r, "boxes", None) is not None and hasattr(r.boxes, "cls") else None
    scores = r.boxes.conf.cpu().numpy() if getattr(r, "boxes", None) is not None and hasattr(r.boxes, "conf") else None

    masks = r.masks.data.cpu().numpy() if getattr(r, "masks", None) is not None else None
    mask_polys = r.masks.xy if getattr(r, "masks", None) is not None and hasattr(r.masks, "xy") else None

    n_boxes = len(boxes) if boxes is not None else 0
    n_masks = len(masks) if masks is not None else 0
    count = max(n_boxes, n_masks)

    for i in range(count):
        b = None
        if boxes is not None and i < len(boxes):
            b = boxes[i]
        elif masks is not None and i < len(masks):
            yy, xx = np.where(masks[i] > 0.5)
            if len(xx) > 0 and len(yy) > 0:
                b = np.array([xx.min(), yy.min(), xx.max(), yy.max()], dtype=np.float32)

        if b is None:
            continue

        c = int(classes[i]) if classes is not None and i < len(classes) else -1
        s = float(scores[i]) if scores is not None and i < len(scores) else 1.0
        label = names.get(c, "segment") if isinstance(names, dict) else "segment"

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


def parse_text_prompt(options: Dict[str, Any]) -> List[str]:
    raw = str(options.get("text_prompt", "")).strip()
    if not raw:
        return []
    return [x.strip().lower() for x in raw.split(",") if x.strip()]


def derive_prompt_bboxes_from_text(image: Image.Image, prompts: List[str], conf: float, imgsz: int) -> List[List[float]]:
    if not prompts:
        return []

    model = get_detection_model()
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    results = model.predict(source=bgr, imgsz=imgsz, conf=conf, verbose=False)
    if not results:
        return []

    r = results[0]
    if r.boxes is None:
        return []

    names = getattr(r, "names", {}) if isinstance(getattr(r, "names", {}), dict) else {}
    selected_cls: set[int] = set()
    for class_id, class_name in names.items():
        n = str(class_name).lower().strip()
        if any((p in n) or (n in p) for p in prompts):
            selected_cls.add(int(class_id))

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    out: List[List[float]] = []
    for i in range(min(len(boxes_xyxy), len(cls_ids))):
        if cls_ids[i] in selected_cls:
            b = boxes_xyxy[i].tolist()
            out.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
            if len(out) >= 24:
                break
    return out


def run_sam2_image(image: Image.Image, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    threshold = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    imgsz = safe_int(options.get("imgsz", 1024), 1024, 320, 2048)
    text_prompts = parse_text_prompt(options)

    kwargs: Dict[str, Any] = {}
    for key in ("bboxes", "points", "labels", "masks"):
        if key in options and options[key] is not None:
            kwargs[key] = options[key]

    if "bboxes" not in kwargs and "points" not in kwargs and text_prompts:
        auto_bboxes = derive_prompt_bboxes_from_text(image, text_prompts, conf=threshold, imgsz=min(1280, imgsz))
        if auto_bboxes:
            kwargs["bboxes"] = auto_bboxes

    img_tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img_path = img_tmp.name
    img_tmp.close()

    try:
        image.save(img_path, format="JPEG")
        model = get_sam2_model(model_id)
        results = model.predict(source=img_path, conf=threshold, imgsz=imgsz, verbose=False, **kwargs)
        if not results:
            return {"instances": [], "concepts": text_prompts}
        return {"instances": extract_instances_from_result(results[0]), "concepts": text_prompts}
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)


def run_sam2_video(payload_base64: str, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    threshold = safe_float(options.get("threshold", 0.25), 0.25, 0.01, 0.95)
    max_frames = safe_int(options.get("max_frames", 90), 90, 4, 300)
    imgsz = safe_int(options.get("imgsz", 1024), 1024, 320, 2048)
    text_prompts = parse_text_prompt(options)

    in_path = decode_video_to_tempfile(payload_base64)
    cap = cv2.VideoCapture(in_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    first_frame_ok, first_frame = cap.read()
    cap.release()

    try:
        kwargs: Dict[str, Any] = {}
        for key in ("points", "labels", "bboxes", "masks", "obj_ids"):
            if key in options and options[key] is not None:
                kwargs[key] = options[key]

        if "bboxes" not in kwargs and "points" not in kwargs and text_prompts and first_frame_ok:
            first_img = Image.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            auto_bboxes = derive_prompt_bboxes_from_text(first_img, text_prompts, conf=threshold, imgsz=min(1280, imgsz))
            if auto_bboxes:
                kwargs["bboxes"] = auto_bboxes

        has_visual_prompts = any(k in kwargs for k in ("points", "bboxes", "masks"))
        frames: List[np.ndarray] = []
        total_instances = 0

        if has_visual_prompts:
            try:
                from ultralytics.models.sam import SAM2VideoPredictor  # type: ignore
            except Exception as exc:
                raise RuntimeError("SAM2VideoPredictor unavailable. Install ultralytics with SAM2 support.") from exc

            predictor = SAM2VideoPredictor(
                overrides={
                    "conf": threshold,
                    "task": "segment",
                    "mode": "predict",
                    "imgsz": imgsz,
                    "model": model_id,
                    "verbose": False,
                }
            )

            stream = predictor(source=in_path, stream=True, **kwargs)
            for idx, r in enumerate(stream):
                frames.append(r.plot())
                total_instances += len(extract_instances_from_result(r))
                if idx + 1 >= max_frames:
                    break
            run_mode = "video_predictor"
        else:
            # Some Ultralytics SAM2 video predictor builds require points/boxes.
            # Fallback: process video frame-by-frame with SAM2 image predictor.
            sam_model = get_sam2_model(model_id)
            cap2 = cv2.VideoCapture(in_path)
            idx = 0
            while idx < max_frames:
                ok, frame = cap2.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = sam_model.predict(source=rgb, conf=threshold, imgsz=imgsz, verbose=False)
                if results:
                    r = results[0]
                    frames.append(r.plot())
                    total_instances += len(extract_instances_from_result(r))
                else:
                    frames.append(frame)
                idx += 1
            cap2.release()
            run_mode = "framewise_fallback"

        if not frames:
            raise RuntimeError("SAM2 video segmentation returned no frames.")

        return {
            "annotated_video": encode_video_base64(frames, fps=float(source_fps or 10.0)),
            "content_type": "video/mp4",
            "concepts": text_prompts,
            "metrics": {
                "frames_processed": len(frames),
                "instances_detected_total": int(total_instances),
                "fps_source": float(source_fps or 10.0),
                "mode": run_mode,
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


def run_velocity_estimation(payload_base64: str, model_id: str, options: Dict[str, Any]) -> Dict[str, Any]:
    in_path = decode_video_to_tempfile(payload_base64)

    max_frames = safe_int(options.get("max_frames", 180), 180, 4, 600)
    meter_per_pixel = safe_float(options.get("meter_per_pixel", 0.05), 0.05, 0.0001, 10.0)
    max_hist = safe_int(options.get("max_hist", 5), 5, 2, 30)
    max_speed = safe_int(options.get("max_speed", 120), 120, 10, 500)
    conf = safe_float(options.get("threshold", 0.1), 0.1, 0.01, 0.95)
    iou = safe_float(options.get("iou", 0.7), 0.7, 0.05, 0.95)
    imgsz = safe_int(options.get("imgsz", 640), 640, 320, 1280)
    tracker = str(options.get("tracker", "botsort.yaml"))
    classes = safe_list_of_ints(options.get("classes"))

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Unable to open uploaded video.")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps_for_calc = safe_float(options.get("fps", source_fps), source_fps, 1.0, 240.0)

    speed_estimator_args: Dict[str, Any] = {
        "model": model_id,
        "fps": fps_for_calc,
        "max_hist": max_hist,
        "meter_per_pixel": meter_per_pixel,
        "max_speed": max_speed,
        "tracker": tracker,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "show": False,
        "verbose": False,
        "device": options.get("device") or get_device(),
    }
    if classes is not None:
        speed_estimator_args["classes"] = classes

    estimator = solutions.SpeedEstimator(**speed_estimator_args)

    frames: list[np.ndarray] = []
    speed_values: list[float] = []
    last_track_speeds: Dict[str, float] = {}

    frame_count = 0
    while frame_count < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        processed = estimator(frame)
        annotated = getattr(processed, "plot_im", None)
        if annotated is None:
            annotated = getattr(estimator, "plot_im", None)
        if annotated is None:
            annotated = frame
        frames.append(annotated)

        track_speeds: Dict[str, float] = {}
        for attr in ("spds", "speeds", "speed", "track_speeds"):
            value = getattr(estimator, attr, None)
            if isinstance(value, dict):
                for k, v in value.items():
                    try:
                        track_speeds[str(k)] = float(v)
                    except Exception:
                        continue
                break

        last_track_speeds = track_speeds
        speed_values.extend(track_speeds.values())

        frame_count += 1

    cap.release()

    if not frames:
        raise RuntimeError("Speed estimation produced no output frames.")

    mean_speed = float(np.mean(speed_values)) if speed_values else 0.0
    max_observed_speed = float(np.max(speed_values)) if speed_values else 0.0

    return {
        "annotated_video": encode_video_base64(frames, fps=source_fps),
        "content_type": "video/mp4",
        "speed": {
            "fps": fps_for_calc,
            "meter_per_pixel": meter_per_pixel,
            "track_speeds": [{"track_id": k, "speed": v} for k, v in last_track_speeds.items()],
        },
        "metrics": {
            "frames_processed": len(frames),
            "fps_source": source_fps,
            "fps_used": fps_for_calc,
            "meter_per_pixel": meter_per_pixel,
            "tracked_objects_last_frame": len(last_track_speeds),
            "mean_speed": mean_speed,
            "max_speed_observed": max_observed_speed,
            "max_speed_config": max_speed,
        },
    }


app = FastAPI(title="Perception Concept Studio Backend", version="2.2.0")

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
            return run_velocity_estimation(request.payloadBase64, model_id, request.options)
        if task == "sam2-segmentation":
            if (request.mimeType or "").lower().startswith("video/"):
                return run_sam2_video(request.payloadBase64, model_id, request.options)
            image = decode_image(request.payloadBase64)
            return run_sam2_image(image, model_id, request.options)

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
