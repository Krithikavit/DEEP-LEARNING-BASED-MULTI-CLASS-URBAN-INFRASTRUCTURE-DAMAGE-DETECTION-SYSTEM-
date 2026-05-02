"""
RDD YOLO detector — runs our trained YOLOv8s on arbitrary road photos
or satellite tiles to detect damage (cracks, potholes).

Trained on the RDD2022 India subset (~2656 images, mAP50 0.357).
Strongest on alligator crack (0.63) and pothole (0.39).

Used by the "click anywhere on map" and "upload pavement" flows.
VIT Campus mode uses STCrackNet + binary classifier and does NOT call this.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Class-color map for drawing boxes.  Roughly aligned with severity:
# potholes are worst (red), alligator (orange), long/trans (yellow/cyan).
CLASS_COLORS = {
    "Longitudinal Crack": (255, 210, 80),    # amber
    "Transverse Crack":   (100, 210, 240),   # cyan
    "Alligator Crack":    (255, 150, 60),    # orange
    "Pothole":            (255, 80, 120),    # magenta-red
}
DEFAULT_COLOR = (200, 200, 200)


def load_rdd_model(weights_path: str):
    """Load the trained YOLO model.  Returns None if weights missing."""
    if not Path(weights_path).exists():
        return None
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except Exception as e:
        print(f"[WARN] RDD YOLO load failed: {e}")
        return None


def detect_damage(model, bgr: np.ndarray, conf_threshold: float = 0.20,
                  iou_threshold: float = 0.45) -> dict:
    """
    Run YOLO inference on an image.  Returns a dict with:
      - detections: list of {class_name, class_id, conf, box [x1,y1,x2,y2]}
      - counts: {class_name: count} summary
      - max_conf: highest-confidence detection
      - rdi: 0-100 severity score derived from detections
    """
    if model is None:
        return {
            "detections": [],
            "counts": {},
            "max_conf": 0.0,
            "rdi": 0.0,
            "dominant_class": None,
            "classification": "NORMAL",
        }

    try:
        results = model(bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    except Exception as e:
        print(f"[WARN] YOLO inference failed: {e}")
        return {
            "detections": [], "counts": {}, "max_conf": 0.0,
            "rdi": 0.0, "dominant_class": None, "classification": "NORMAL",
        }

    r = results[0]
    boxes = r.boxes

    detections = []
    counts: dict[str, int] = {}
    max_conf = 0.0

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

    # Severity derivation — rough but reasonable:
    #   - Potholes are worst  (each +20 to RDI)
    #   - Alligator crack       (each +15)
    #   - Transverse/Longitudinal (each +8)
    # Weighted by confidence.  Capped at 100.
    weights = {
        "Pothole": 20.0,
        "Alligator Crack": 15.0,
        "Transverse Crack": 8.0,
        "Longitudinal Crack": 8.0,
    }
    raw = sum(weights.get(d["class_name"], 5.0) * d["conf"] for d in detections)
    rdi = float(min(round(raw, 2), 100.0))

    # Classification from RDI bands
    if rdi < 10:
        classification = "NORMAL"
    elif rdi < 30:
        classification = "MINOR"
    elif rdi < 60:
        classification = "MODERATE"
    else:
        classification = "SEVERE"

    # Dominant class = the class with the most total weighted confidence
    class_weights: dict[str, float] = {}
    for d in detections:
        class_weights[d["class_name"]] = \
            class_weights.get(d["class_name"], 0.0) + d["conf"]
    dominant = max(class_weights, key=class_weights.get) if class_weights else None

    return {
        "detections": detections,
        "counts": counts,
        "max_conf": round(max_conf, 4),
        "rdi": rdi,
        "dominant_class": dominant,
        "classification": classification,
    }


def render_detections(bgr: np.ndarray, detections: list) -> np.ndarray:
    """Draw bounding boxes + labels on a copy of the image."""
    out = bgr.copy()
    h, w = out.shape[:2]
    thickness = max(2, int(min(w, h) / 400))
    font_scale = max(0.5, min(w, h) / 1400)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls_name = det["class_name"]
        conf = det["conf"]
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
        # OpenCV is BGR; our colors are RGB-ish, swap for draw:
        color_bgr = (color[2], color[1], color[0])

        cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, thickness)

        # Label box with confidence
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 1)
        # background rect for text
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 8, y1),
                      color_bgr, -1)
        cv2.putText(out, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 30), 1,
                    cv2.LINE_AA)

    return out


def analyze_with_rdd(model, bgr: np.ndarray,
                     conf_threshold: float = 0.20) -> dict:
    """
    Convenience wrapper: runs detection + renders annotated output image.
    Returns a dict compatible with the existing analysis pipeline format.
    """
    result = detect_damage(model, bgr, conf_threshold=conf_threshold)
    annotated = render_detections(bgr, result["detections"])
    ok, buf = cv2.imencode(".png", annotated)
    overlay_png = buf.tobytes() if ok else b""

    ok, buf = cv2.imencode(".png", bgr)
    input_png = buf.tobytes() if ok else b""

    return {
        "mode": "rdd_detection",
        "model": "YOLOv8s (RDD2022 India, our training)",
        "metrics": {
            "n_detections": len(result["detections"]),
            "counts": result["counts"],
            "max_conf": result["max_conf"],
            "rdi": result["rdi"],
            "classification": result["classification"],
            "dominant_class": result["dominant_class"],
            "detections": result["detections"],
        },
        "input_png": input_png,
        "overlay_png": overlay_png,
    }
