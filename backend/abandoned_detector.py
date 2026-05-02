"""
Abandoned Building Detector — loads a YOLOv8 model trained on the xBD
dataset (normal_building vs abandoned_building) for the teammate's
contribution to the shared RoadSense dashboard.

Classes:
  0 → normal_building
  1 → abandoned_building

Input: any BGR image (satellite tile or uploaded photo).
Output: bounding boxes with class + confidence, plus a verdict
(NORMAL / ABANDONED / MIXED) for the frame as a whole.
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


# Class colors for drawing boxes (BGR internally, RGB source below)
CLASS_COLORS = {
    "normal_building":    (100, 220, 100),   # green
    "abandoned_building": (80, 80, 255),     # red
}
DEFAULT_COLOR = (200, 200, 200)


def load_abandoned_model(weights_path: str):
    """Load the teammate's trained YOLO model. Returns None if missing."""
    if not Path(weights_path).exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(weights_path)
    except Exception as e:
        print(f"[WARN] Abandoned-building model load failed: {e}")
        return None


def detect_buildings(model, bgr: np.ndarray,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> dict:
    """
    Run the abandoned-building detector on an image.

    Returns:
      detections: list of {class_name, class_id, conf, box}
      counts: {class_name: count}
      max_conf: highest-confidence detection
      verdict: frame-level verdict NORMAL / ABANDONED / MIXED / NO_BUILDINGS
      abandoned_ratio: fraction of detections classified as abandoned
    """
    if model is None:
        return {
            "detections": [],
            "counts": {},
            "max_conf": 0.0,
            "verdict": "MODEL_NOT_LOADED",
            "abandoned_ratio": 0.0,
        }

    try:
        results = model(bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    except Exception as e:
        print(f"[WARN] Abandoned-building inference failed: {e}")
        return {
            "detections": [], "counts": {}, "max_conf": 0.0,
            "verdict": "INFERENCE_FAILED", "abandoned_ratio": 0.0,
        }

    r = results[0]
    boxes = r.boxes

    detections = []
    counts: dict[str, int] = {}
    max_conf = 0.0
    abandoned_count = 0
    normal_count = 0
    # Confidence-weighted counts, more honest than raw counts:
    abandoned_weight = 0.0
    normal_weight = 0.0

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class_name": cls_name,
                "class_id": cls_id,
                "conf": round(conf, 4),
                "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })
            counts[cls_name] = counts.get(cls_name, 0) + 1
            max_conf = max(max_conf, conf)

            if cls_name == "abandoned_building":
                abandoned_count += 1
                abandoned_weight += conf
            elif cls_name == "normal_building":
                normal_count += 1
                normal_weight += conf

    # Decide overall verdict
    total_detections = abandoned_count + normal_count
    if total_detections == 0:
        verdict = "NO_BUILDINGS"
        abandoned_ratio = 0.0
    else:
        abandoned_ratio = abandoned_weight / (abandoned_weight + normal_weight + 1e-6)
        if abandoned_ratio >= 0.60:
            verdict = "ABANDONED"
        elif abandoned_ratio <= 0.20:
            verdict = "NORMAL"
        else:
            verdict = "MIXED"

    return {
        "detections": detections,
        "counts": counts,
        "max_conf": round(max_conf, 4),
        "verdict": verdict,
        "abandoned_ratio": round(abandoned_ratio, 4),
        "n_detections": len(detections),
        "n_abandoned": abandoned_count,
        "n_normal": normal_count,
    }


def render_detections(bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + class labels on a copy of the image."""
    out = bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(min(w, h) / 400))
    font_scale = max(0.5, min(w, h) / 1400)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_name = det["class_name"]
        conf = det["conf"]
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        label_text = "NORMAL" if cls_name == "normal_building" else "ABANDONED"
        label = f"{label_text} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 8, y1),
                      color, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), 1,
                    cv2.LINE_AA)

    return out


def analyze_building_image(model, bgr: np.ndarray,
                           conf_threshold: float = 0.25) -> dict:
    """
    Convenience wrapper: runs detection + renders annotated output.
    Returns a dict compatible with the dashboard's rendering format.
    """
    result = detect_buildings(model, bgr, conf_threshold=conf_threshold)

    annotated = render_detections(bgr, result["detections"])
    ok, buf = cv2.imencode(".png", annotated)
    overlay_png = buf.tobytes() if ok else b""

    ok, buf = cv2.imencode(".png", bgr)
    input_png = buf.tobytes() if ok else b""

    return {
        "mode": "building_detection",
        "model": "YOLOv8m — Abandoned Building Detector (xBD)",
        "metrics": {
            "verdict": result["verdict"],
            "n_detections": result["n_detections"],
            "n_normal": result["n_normal"],
            "n_abandoned": result["n_abandoned"],
            "abandoned_ratio": result["abandoned_ratio"],
            "max_conf": result["max_conf"],
            "counts": result["counts"],
            "detections": result["detections"],
        },
        "input_png": input_png,
        "overlay_png": overlay_png,
    }
