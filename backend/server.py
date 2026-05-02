"""
RoadSense API — FastAPI server v3.1 (with Telegram alerts).

Changes from v3:
  - Telegram alerts on DAMAGED/SEVERE road clicks (satellite)
  - Telegram alerts on DAMAGED/SEVERE pavement uploads
  - Telegram alerts on ABANDONED/MIXED building detections
  - Alert includes verdict, lat/lon, confidence, map link, AND photo of tile
  - All alerts are best-effort: failures logged, NEVER break the response
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from .analysis import analyze_pavement, analyze_satellite
from .providers import fetch_tile, grid_points

# >>> NEW: telegram alerts <<<
try:
    from .telegram_alert import send_alert as _send_telegram_alert
    _TELEGRAM_OK = True
    print("[OK] Telegram alerter loaded.")
except Exception as _e:
    _TELEGRAM_OK = False
    print(f"[WARN] Telegram alerter not available: {_e}")

    def _send_telegram_alert(*args, **kwargs):
        return False


DEVICE = os.environ.get("ROADSENSE_DEVICE", "cpu")
WEIGHTS_PATH = os.environ.get(
    "ROADSENSE_WEIGHTS",
    str(Path(__file__).parent.parent / "weights" / "STCrackNet_final.pth"),
)
SEVERITY_WEIGHTS = os.environ.get(
    "VIT_SEVERITY_WEIGHTS",
    str(Path(__file__).parent.parent / "weights" / "STCrackNet_vit_severity.pth"),
)
RDD_WEIGHTS = os.environ.get(
    "RDD_YOLO_WEIGHTS",
    str(Path(__file__).parent.parent / "weights" / "rdd_india_yolo.pt"),
)
ABANDONED_WEIGHTS = os.environ.get(
    "ABANDONED_BUILDING_WEIGHTS",
    str(Path(__file__).parent.parent / "weights" / "abandoned_building.pt"),
)
CAMPUS_JSON = Path(__file__).parent.parent / "data" / "campus.json"
CAMPUS_PHOTOS_DIR = Path(__file__).parent.parent / "data" / "vit_photos"

app = FastAPI(title="RoadSense", version="3.1")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

state: dict = {
    "model": None,
    "model_ok": False,
    "severity_model": None,
    "severity_class_names": None,
    "severity_model_ok": False,
    "rdd_model": None,
    "rdd_model_ok": False,
    "abandoned_model": None,
    "abandoned_model_ok": False,
    "history": deque(maxlen=200),
    "images": {},
    "campus": None,
    "campus_results": {},
    "campus_features": {},
}


def _store_images(result: dict) -> str:
    rid = uuid.uuid4().hex[:12]
    payload = {}
    for key, val in result.items():
        if key.endswith("_png"):
            payload[key.replace("_png", "")] = val
    state["images"][rid] = payload
    return rid


# >>> NEW: helper to extract the "best" png bytes to attach to an alert <<<
def _pick_alert_image(result: dict) -> bytes | None:
    """
    Choose the most informative PNG to send with the alert.
    Preference: overlay > rdd_overlay > land > input.
    """
    for k in ("overlay_png", "rdd_overlay_png", "land_png", "input_png"):
        if result.get(k):
            return result[k]
    return None


# >>> NEW: best-effort alert with logging, never raises <<<
def _fire_alert(
    *,
    title: str,
    lat: float,
    lon: float,
    verdict: str,
    confidence: float | None,
    image_bytes: bytes | None,
    extra_info: dict | None = None,
) -> None:
    if not _TELEGRAM_OK:
        return
    try:
        ok = _send_telegram_alert(
            title=title,
            lat=float(lat),
            lon=float(lon),
            verdict=str(verdict),
            confidence=(float(confidence) if confidence is not None else None),
            image_bytes=image_bytes,
            extra_info=extra_info or {},
        )
        if ok:
            print(f"[alert] sent: {title} @ {lat:.4f},{lon:.4f} ({verdict})")
    except Exception as e:
        print(f"[alert] skipped: {e}")


def _preload_campus():
    """VIT Campus — STCrackNet + binary severity classifier.  UNTOUCHED."""
    if not CAMPUS_JSON.exists():
        print(f"[INFO] No campus.json at {CAMPUS_JSON} — campus mode disabled.")
        return

    try:
        with open(CAMPUS_JSON) as f:
            campus = json.load(f)
    except Exception as e:
        print(f"[WARN] campus.json load failed: {e}")
        return

    state["campus"] = campus
    total = len(campus.get("photos", []))
    print(f"[INFO] Campus catalog: {campus.get('campus','?')} — {total} photos")

    if not state["model_ok"]:
        print("[WARN] STCrackNet not loaded — campus disabled.")
        return

    for i, photo in enumerate(campus["photos"], 1):
        fpath = CAMPUS_PHOTOS_DIR / photo["filename"]
        if not fpath.exists():
            print(f"[WARN] {photo['id']}: file not found")
            continue
        try:
            bgr = cv2.imread(str(fpath))
            if bgr is None:
                print(f"[WARN] {photo['id']}: could not decode")
                continue

            result = analyze_pavement(state["model"], bgr, DEVICE)

            used_model = "STCrackNet"
            if state["severity_model_ok"]:
                from .severity import predict_severity, extract_features
                sev = predict_severity(
                    state["severity_model"], state["severity_class_names"],
                    bgr, DEVICE, use_tta=True,
                )
                result["metrics"]["classification"] = sev["class_name"]
                result["metrics"]["rdi"] = sev["rdi"]
                result["metrics"]["confidence"] = sev["confidence"]
                result["metrics"]["probabilities"] = sev["probabilities"]
                result["metrics"]["tta_disagreement"] = sev["tta_disagreement"]
                used_model = "STCrackNet + VIT Severity (TTA)"

                feat = extract_features(state["severity_model"], bgr, DEVICE)
                state["campus_features"][photo["id"]] = feat

            rid = _store_images(result)
            state["campus_results"][photo["id"]] = {
                "id": rid,
                "photo_id":  photo["id"],
                "location":  photo["location"],
                "lat":       photo["lat"],
                "lon":       photo["lon"],
                "metrics":   result["metrics"],
                "expected":  photo.get("expected_severity"),
                "mode":      "pavement",
                "model":     used_model,
                "image_urls": {
                    "input":   f"/api/image/{rid}/input",
                    "mask":    f"/api/image/{rid}/mask",
                    "overlay": f"/api/image/{rid}/overlay",
                },
            }
            print(f"  [{i}/{total}] {photo['id']}: "
                  f"{result['metrics']['classification']} "
                  f"(RDI {result['metrics']['rdi']}, "
                  f"conf {result['metrics'].get('confidence', 0):.2f}) "
                  f"— {photo['location']}")
        except Exception as e:
            print(f"[WARN] {photo['id']}: failed — {e}")

    print(f"[OK] Precomputed {len(state['campus_features'])} feature vectors for retrieval")


@app.on_event("startup")
def _startup():
    # 1. STCrackNet (VIT Campus segmentation overlay)
    if Path(WEIGHTS_PATH).exists():
        try:
            from .model import load_stcracknet
            state["model"] = load_stcracknet(WEIGHTS_PATH, DEVICE)
            state["model_ok"] = True
            print(f"[OK] STCrackNet loaded from {WEIGHTS_PATH} on {DEVICE}")
        except Exception as e:
            print(f"[WARN] STCrackNet load failed: {e}")
    else:
        print(f"[WARN] STCrackNet weights not found at {WEIGHTS_PATH}")

    # 2. VIT severity classifier (binary DAMAGED/SEVERE)
    if Path(SEVERITY_WEIGHTS).exists():
        try:
            from .severity import load_severity_model
            sm, class_names = load_severity_model(SEVERITY_WEIGHTS, DEVICE)
            if sm is not None:
                state["severity_model"] = sm
                state["severity_class_names"] = class_names
                state["severity_model_ok"] = True
                print(f"[OK] VIT severity classifier loaded "
                      f"({len(class_names)} classes: {', '.join(class_names)})")
            else:
                print(f"[WARN] VIT severity classifier file exists but failed to load.")
        except Exception as e:
            print(f"[WARN] VIT severity load failed: {e}")
    else:
        print(f"[INFO] VIT severity classifier not found at {SEVERITY_WEIGHTS}")

    # 3. RDD YOLO (for click-anywhere / upload detection)
    if Path(RDD_WEIGHTS).exists():
        try:
            from .rdd_detector import load_rdd_model
            rm = load_rdd_model(RDD_WEIGHTS)
            if rm is not None:
                state["rdd_model"] = rm
                state["rdd_model_ok"] = True
                print(f"[OK] RDD YOLO loaded ({len(rm.names)} classes: "
                      f"{', '.join(rm.names.values())})")
            else:
                print(f"[WARN] RDD YOLO weights exist but failed to load")
        except Exception as e:
            print(f"[WARN] RDD YOLO load failed: {e}")
    else:
        print(f"[INFO] RDD YOLO weights not found at {RDD_WEIGHTS}")
        print(f"       Click-anywhere / satellite detection will be unavailable.")

    # 4. Abandoned-Building detector (teammate's model, YOLOv8m)
    if Path(ABANDONED_WEIGHTS).exists():
        try:
            from .abandoned_detector import load_abandoned_model
            am = load_abandoned_model(ABANDONED_WEIGHTS)
            if am is not None:
                state["abandoned_model"] = am
                state["abandoned_model_ok"] = True
                print(f"[OK] Abandoned-building model loaded "
                      f"({len(am.names)} classes: {', '.join(am.names.values())})")
            else:
                print(f"[WARN] Abandoned-building weights exist but failed to load")
        except Exception as e:
            print(f"[WARN] Abandoned-building load failed: {e}")
    else:
        print(f"[INFO] Abandoned-building weights not found at {ABANDONED_WEIGHTS}")
        print(f"       Building-analysis mode will be unavailable.")

    _preload_campus()


# ─── Request models ─────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    provider: str = Field("auto", pattern="^(auto|google|sentinel)$")
    zoom: int = Field(19, ge=10, le=21)


class RegionRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    half_side_m: float = Field(200.0, gt=10, le=5000)
    grid_n: int = Field(3, ge=1, le=5)
    provider: str = Field("auto", pattern="^(auto|google|sentinel)$")
    zoom: int = Field(19, ge=10, le=21)


# ─── Helpers ────────────────────────────────────────────────

def _record_analysis(lat: float, lon: float, result: dict) -> dict:
    rid = _store_images(result)
    record = {
        "id": rid, "lat": lat, "lon": lon,
        "timestamp": time.time(),
        "metrics": result["metrics"],
        "mode": result["mode"],
        "model": result["model"],
    }
    state["history"].appendleft(record)
    return record


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "stcracknet_loaded": state["model_ok"],
        "severity_classifier_loaded": state["severity_model_ok"],
        "severity_classes": state["severity_class_names"],
        "rdd_yolo_loaded": state["rdd_model_ok"],
        "rdd_classes": (list(state["rdd_model"].names.values())
                         if state["rdd_model_ok"] else None),
        "abandoned_model_loaded": state["abandoned_model_ok"],
        "abandoned_classes": (list(state["abandoned_model"].names.values())
                               if state["abandoned_model_ok"] else None),
        "telegram_alerts": _TELEGRAM_OK,
        "providers": {
            "google": bool(os.environ.get("GOOGLE_MAPS_API_KEY")),
            "sentinel": bool(os.environ.get("SENTINEL_CLIENT_ID")),
        },
        "campus": {
            "loaded": state["campus"] is not None,
            "name": state["campus"].get("campus") if state["campus"] else None,
            "photos_analyzed": len(state["campus_results"]),
            "feature_vectors": len(state["campus_features"]),
        },
    }


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    """
    Click-anywhere-on-map analysis.
    """
    try:
        tile = fetch_tile(req.lat, req.lon, provider=req.provider, zoom=req.zoom)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"tile fetch failed: {e}")

    # 1. Land-cover classification
    result = analyze_satellite(tile.bgr)
    metrics = result["metrics"]
    road_frac = float(metrics.get("road_coverage", 0.0))

    # 2. Road-gated damage assessment using 5-feature visual analysis.
    ROAD_THRESHOLD = 0.12
    dominant = metrics.get("dominant_cover", "unknown")
    road_dominates = (dominant == "road")
    if road_frac >= ROAD_THRESHOLD and road_dominates:
        try:
            from .analysis import extract_road_mask
            from .road_quality import assess_road_quality

            road_mask = extract_road_mask(tile.bgr)
            assessment = assess_road_quality(tile.bgr, road_mask)

            if assessment["classification"] is not None:
                metrics["classification"] = assessment["classification"]
                metrics["rdi"] = assessment["rdi"]
                metrics["confidence"] = assessment["confidence"]
                metrics["probabilities"] = assessment["probabilities"]
                metrics["road_features"] = assessment["features"]
                metrics["damage_score"] = assessment["damage_score"]
                metrics["assessment_method"] = "visual_feature_analysis"
            else:
                metrics["classification"] = None
                metrics["assessment_method"] = "too_little_road"
                metrics["assessment_reason"] = assessment.get("reason", "")
        except Exception as e:
            print(f"[WARN] road quality assessment failed: {e}")
            import traceback; traceback.print_exc()
            metrics["classification"] = None
            metrics["assessment_method"] = "failed"
    else:
        metrics["classification"] = None
        metrics["rdi"] = None
        metrics["assessment_method"] = "not_a_road"
        if road_frac < ROAD_THRESHOLD:
            metrics["assessment_reason"] = (
                f"Road coverage {road_frac*100:.0f}% < {ROAD_THRESHOLD*100:.0f}% threshold"
            )
        else:
            metrics["assessment_reason"] = (
                f"{dominant.capitalize()} dominates ({metrics.get(dominant + '_coverage', 0)*100:.0f}%) — road is only {road_frac*100:.0f}%"
            )

    record = _record_analysis(req.lat, req.lon, result)

    # >>> NEW: fire Telegram alert if DAMAGED or SEVERE <<<
    verdict = metrics.get("classification")
    if verdict in ("DAMAGED", "SEVERE"):
        _fire_alert(
            title=f"{verdict} Road Detected (satellite click)",
            lat=req.lat, lon=req.lon,
            verdict=verdict,
            confidence=metrics.get("confidence"),
            image_bytes=_pick_alert_image(result),
            extra_info={
                "damage_score": metrics.get("damage_score"),
                "rdi": metrics.get("rdi"),
                "dominant_cover": metrics.get("dominant_cover"),
                "road_coverage": f"{road_frac*100:.0f}%",
                "method": metrics.get("assessment_method"),
            },
        )

    return {
        **record,
        "provider": tile.provider,
        "meters_per_pixel": tile.meters_per_pixel,
        "image_urls": {
            "input": f"/api/image/{record['id']}/input",
            "land":  f"/api/image/{record['id']}/land",
        },
    }


@app.post("/api/analyze-region")
def analyze_region(req: RegionRequest):
    """
    Region scan — samples a grid of tiles.
    """
    import concurrent.futures as _cf

    n = max(req.grid_n, 3)
    points = grid_points(req.lat, req.lon, req.half_side_m, n)

    has_severity = state["severity_model_ok"]

    def _one(p):
        plat, plon = p
        try:
            tile = fetch_tile(plat, plon, provider=req.provider, zoom=req.zoom)
            result = analyze_satellite(tile.bgr)
            metrics = result["metrics"]

            if has_severity:
                try:
                    from .severity import predict_severity
                    sev = predict_severity(
                        state["severity_model"],
                        state["severity_class_names"],
                        tile.bgr, DEVICE, use_tta=False,
                    )
                    metrics["classification"] = sev["class_name"]
                    metrics["rdi"] = sev["rdi"]
                    metrics["confidence"] = sev["confidence"]
                    metrics["probabilities"] = sev["probabilities"]
                except Exception:
                    pass

            record = _record_analysis(plat, plon, result)
            urls = {
                "input": f"/api/image/{record['id']}/input",
                "land":  f"/api/image/{record['id']}/land",
            }
            return {**record, "image_urls": urls}
        except Exception as e:
            return {"lat": plat, "lon": plon, "error": str(e)}

    with _cf.ThreadPoolExecutor(max_workers=min(len(points), 4)) as ex:
        all_points = list(ex.map(_one, points))

    segments = [s for s in all_points if "error" not in s]
    errors   = [s for s in all_points if "error" in s]

    cls_counts = {"DAMAGED": 0, "SEVERE": 0, "NORMAL": 0}
    cover_counts = {"road": 0, "building": 0, "vegetation": 0,
                    "bare ground": 0, "unknown": 0}
    for s in segments:
        c = s["metrics"].get("classification") or "NORMAL"
        cls_counts[c] = cls_counts.get(c, 0) + 1
        dom = s["metrics"].get("dominant_cover", "unknown")
        cover_counts[dom] = cover_counts.get(dom, 0) + 1

    return {
        "center": {"lat": req.lat, "lon": req.lon},
        "half_side_m": req.half_side_m,
        "segments": segments,
        "summary": {
            "total": len(segments),
            "errors": len(errors),
            "by_classification": cls_counts,
            "by_cover": cover_counts,
        },
    }


@app.post("/api/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    """
    Upload a photo: runs BOTH STCrackNet (with severity + retrieval) AND
    RDD YOLO bounding-box detection.
    """
    if not state["model_ok"]:
        raise HTTPException(
            status_code=503, detail="STCrackNet weights not loaded.",
        )

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # STCrackNet segmentation overlay
    result = analyze_pavement(state["model"], bgr, DEVICE)

    # Severity (TTA)
    similar_photos = []
    if state["severity_model_ok"]:
        from .severity import predict_severity, extract_features, retrieve_similar
        sev = predict_severity(
            state["severity_model"], state["severity_class_names"],
            bgr, DEVICE, use_tta=True,
        )
        result["metrics"]["classification"] = sev["class_name"]
        result["metrics"]["rdi"] = sev["rdi"]
        result["metrics"]["confidence"] = sev["confidence"]
        result["metrics"]["probabilities"] = sev["probabilities"]
        result["metrics"]["tta_disagreement"] = sev["tta_disagreement"]
        result["model"] = "STCrackNet + VIT Severity (TTA)"

        if state["campus_features"]:
            query_feat = extract_features(state["severity_model"], bgr, DEVICE)
            matches = retrieve_similar(query_feat, state["campus_features"], top_k=3)
            for m in matches:
                pid = m["photo_id"]
                cr = state["campus_results"].get(pid)
                if cr:
                    m["class_name"] = cr["metrics"].get("classification")
                    m["location"] = cr.get("location")
                    m["image_url"] = cr["image_urls"]["input"]
                    m["rdi"] = cr["metrics"].get("rdi")
            similar_photos = matches

    # RDD YOLO bounding-box detection
    rdd_result = None
    if state["rdd_model_ok"]:
        from .rdd_detector import detect_damage, render_detections
        rdd = detect_damage(state["rdd_model"], bgr, conf_threshold=0.20)
        rdd_annotated = render_detections(bgr, rdd["detections"])
        ok, buf = cv2.imencode(".png", rdd_annotated)
        if ok:
            result["rdd_overlay_png"] = buf.tobytes()
        rdd_result = {
            "n_detections": len(rdd["detections"]),
            "counts": rdd["counts"],
            "max_conf": rdd["max_conf"],
            "rdi": rdd["rdi"],
            "classification": rdd["classification"],
            "dominant_class": rdd["dominant_class"],
            "detections": rdd["detections"],
        }
        result["metrics"]["rdd"] = rdd_result

    record = _record_analysis(0.0, 0.0, result)

    # >>> NEW: fire Telegram alert on uploads if DAMAGED or SEVERE <<<
    upload_verdict = result["metrics"].get("classification")
    if upload_verdict in ("DAMAGED", "SEVERE"):
        _fire_alert(
            title=f"{upload_verdict} Pavement Detected (uploaded photo)",
            lat=0.0, lon=0.0,  # no GPS on uploads
            verdict=upload_verdict,
            confidence=result["metrics"].get("confidence"),
            image_bytes=_pick_alert_image(result),
            extra_info={
                "rdi": result["metrics"].get("rdi"),
                "n_rdd_detections": (rdd_result["n_detections"] if rdd_result else 0),
                "rdd_verdict": (rdd_result["classification"] if rdd_result else None),
                "source": "upload",
                "file": file.filename,
            },
        )

    image_urls = {
        "input":   f"/api/image/{record['id']}/input",
        "mask":    f"/api/image/{record['id']}/mask",
        "overlay": f"/api/image/{record['id']}/overlay",
    }
    if state["rdd_model_ok"]:
        image_urls["rdd_overlay"] = f"/api/image/{record['id']}/rdd_overlay"

    return {
        **record,
        "image_urls": image_urls,
        "similar_photos": similar_photos,
        "rdd": rdd_result,
    }


# ─── Building Analysis Mode (teammate's model) ──────────────

@app.post("/api/analyze-building")
def analyze_building(req: AnalyzeRequest):
    """
    Building-mode map click: fetch satellite tile, run teammate's
    abandoned-building YOLO.
    """
    if not state["abandoned_model_ok"]:
        raise HTTPException(
            status_code=503,
            detail="Abandoned-building model not loaded on server.",
        )

    try:
        tile = fetch_tile(req.lat, req.lon, provider=req.provider, zoom=req.zoom)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"tile fetch failed: {e}")

    from .abandoned_detector import analyze_building_image
    result = analyze_building_image(
        state["abandoned_model"], tile.bgr, conf_threshold=0.25,
    )
    record = _record_analysis(req.lat, req.lon, result)

    # >>> NEW: fire Telegram alert on ABANDONED or MIXED buildings <<<
    bld_verdict = result["metrics"].get("classification")
    if bld_verdict in ("ABANDONED", "MIXED"):
        _fire_alert(
            title=f"{bld_verdict} Building Detected (satellite click)",
            lat=req.lat, lon=req.lon,
            verdict=bld_verdict,
            confidence=result["metrics"].get("confidence"),
            image_bytes=_pick_alert_image(result),
            extra_info={
                "n_buildings": result["metrics"].get("n_detections"),
                "abandoned_ratio": result["metrics"].get("abandoned_ratio"),
                "dominant_class": result["metrics"].get("dominant_class"),
            },
        )

    return {
        **record,
        "provider": tile.provider,
        "meters_per_pixel": tile.meters_per_pixel,
        "image_urls": {
            "input":   f"/api/image/{record['id']}/input",
            "overlay": f"/api/image/{record['id']}/overlay",
        },
    }


@app.post("/api/analyze-building-upload")
async def analyze_building_upload(file: UploadFile = File(...)):
    """
    Building-mode upload: runs teammate's abandoned-building YOLO
    on an uploaded photo.
    """
    if not state["abandoned_model_ok"]:
        raise HTTPException(
            status_code=503,
            detail="Abandoned-building model not loaded on server.",
        )

    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    from .abandoned_detector import analyze_building_image
    result = analyze_building_image(
        state["abandoned_model"], bgr, conf_threshold=0.25,
    )
    record = _record_analysis(0.0, 0.0, result)

    # >>> NEW: alert on uploaded buildings too <<<
    bld_verdict = result["metrics"].get("classification")
    if bld_verdict in ("ABANDONED", "MIXED"):
        _fire_alert(
            title=f"{bld_verdict} Building Detected (uploaded photo)",
            lat=0.0, lon=0.0,
            verdict=bld_verdict,
            confidence=result["metrics"].get("confidence"),
            image_bytes=_pick_alert_image(result),
            extra_info={
                "n_buildings": result["metrics"].get("n_detections"),
                "abandoned_ratio": result["metrics"].get("abandoned_ratio"),
                "source": "upload",
                "file": file.filename,
            },
        )

    return {
        **record,
        "image_urls": {
            "input":   f"/api/image/{record['id']}/input",
            "overlay": f"/api/image/{record['id']}/overlay",
        },
    }


# >>> NEW: manual test endpoint for alerts <<<
@app.post("/api/test-alert")
def test_alert():
    """Manual smoke test — fires one alert with fake data."""
    ok = _send_telegram_alert(
        title="TEST ALERT (from /api/test-alert)",
        lat=12.9692,
        lon=79.1559,
        verdict="SEVERE",
        confidence=0.87,
        image_bytes=None,
        extra_info={"source": "manual-test", "env": "dev"},
    )
    return {"sent": bool(ok), "telegram_ok": _TELEGRAM_OK}


@app.get("/api/history")
def history(limit: int = Query(50, ge=1, le=200)):
    return list(state["history"])[:limit]


@app.get("/api/campus")
def campus():
    """VIT Campus — untouched."""
    if not state["campus"]:
        raise HTTPException(status_code=404, detail="No campus catalog configured.")
    return {
        "campus": state["campus"].get("campus", ""),
        "center": state["campus"].get("center"),
        "default_zoom": state["campus"].get("default_zoom", 17),
        "photos": list(state["campus_results"].values()),
        "total_photos": len(state["campus"].get("photos", [])),
        "analyzed": len(state["campus_results"]),
        "model_ok": state["model_ok"],
        "severity_model_ok": state["severity_model_ok"],
    }


@app.get("/api/campus/{photo_id}")
def campus_photo(photo_id: str):
    """VIT Campus single photo — untouched."""
    result = state["campus_results"].get(photo_id)
    if not result:
        raise HTTPException(status_code=404, detail="photo not found")
    return result


@app.get("/api/image/{rid}/{key}")
def get_image(rid: str, key: str):
    bucket = state["images"].get(rid)
    if not bucket or key not in bucket:
        raise HTTPException(status_code=404, detail="image not found")
    return Response(content=bucket[key], media_type="image/png")


# ─── Frontend serving ───────────────────────────────────────

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
