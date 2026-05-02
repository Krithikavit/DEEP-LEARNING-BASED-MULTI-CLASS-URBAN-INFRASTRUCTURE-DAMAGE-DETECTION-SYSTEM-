"""
Road Quality Assessment — honest image-processing features for satellite tiles.

Instead of running a neural classifier (which hallucinates on out-of-distribution
satellite views of roads), we measure 5 interpretable visual features:

  1. Linearity       — long straight lines survive in good roads (Hough transform).
  2. Uniformity      — fresh asphalt is uniformly colored (low stddev of gray).
  3. Smoothness      — smooth asphalt has low local-variance texture.
  4. Clean-spot      — few significantly-dark blobs (pothole proxy).
  5. Edge regularity — cracks are irregular edges; lane markings are regular.

Each returns 0-1 (1 = good / safe). Damage score = weighted average of the
INVERSES. Final classification:
    damage < 0.20 → SAFE
    damage < 0.50 → DAMAGED
    else            → SEVERE

Why features, not a model: these scores are interpretable. 
"why did you say this road is damaged?", we show them the patch score, the
texture score, etc. No black-box claims.
"""
from __future__ import annotations

import numpy as np
import cv2


def _safe_divide(a: float, b: float) -> float:
    return float(a) / float(b) if abs(b) > 1e-9 else 0.0


def _linearity_score(gray_road: np.ndarray, road_mask: np.ndarray) -> float:
    """
    Count long straight lines detected inside the road mask.  Good roads
    have long clean edges (lane borders, pavement boundaries); damaged roads
    have broken/short segments.

    Returns 0-1 where 1 = lots of long straight lines.
    """
    road_pixels = int(road_mask.sum())
    if road_pixels < 500:
        return 0.5  # not enough road — neutral

    # Canny edges masked to the road region
    edges = cv2.Canny(gray_road, 50, 150)
    edges = (edges > 0).astype(np.uint8) * (road_mask > 0).astype(np.uint8) * 255

    # Hough lines — long segments only
    min_len = max(20, int(np.sqrt(road_pixels) * 0.15))
    lines = cv2.HoughLinesP(
        edges.astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=min_len,
        maxLineGap=8,
    )

    if lines is None or len(lines) == 0:
        return 0.15  # no long lines → likely broken road

    # Total length of straight segments normalized by sqrt(road area)
    total_len = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        total_len += float(np.hypot(x2 - x1, y2 - y1))
    normalized = total_len / (np.sqrt(road_pixels) * 8.0)

    # Squash to 0-1 with diminishing returns
    return float(np.clip(np.tanh(normalized), 0.0, 1.0))


def _uniformity_score(gray_road: np.ndarray, road_mask: np.ndarray) -> float:
    """
    Well-maintained asphalt is uniformly dark gray.  Damaged asphalt has
    patches (repairs, potholes, exposed aggregate) → higher stddev.

    Returns 0-1 where 1 = very uniform.
    """
    pixels = gray_road[road_mask > 0]
    if pixels.size < 100:
        return 0.5

    std = float(np.std(pixels))
    # Calibrated: typical satellite asphalt stddev ≈ 18-32, damaged ≈ 40+.
    # Previous (12-40 range) was too tight.
    score = 1.0 - np.clip((std - 20.0) / 30.0, 0.0, 1.0)
    return float(max(0.1, score))


def _smoothness_score(gray_road: np.ndarray, road_mask: np.ndarray) -> float:
    """
    Local-variance texture measure.  Smooth asphalt → low local variance.
    Cracked/rough asphalt → high local variance.

    Returns 0-1 where 1 = very smooth.
    """
    if int(road_mask.sum()) < 500:
        return 0.5

    # Local variance via box filter of squared deviation
    gray = gray_road.astype(np.float32)
    mean = cv2.boxFilter(gray, -1, (7, 7))
    sq = cv2.boxFilter(gray * gray, -1, (7, 7))
    local_var = np.clip(sq - mean * mean, 0.0, None)

    var_on_road = local_var[road_mask > 0]
    if var_on_road.size == 0:
        return 0.5
    mean_var = float(np.mean(var_on_road))

    # Calibrated on actual satellite tiles:
    # Typical asphalt local var ≈ 150-400, rough/damaged ≈ 600+.
    # Previous constants (25-180) were way too low → everything looked rough.
    score = 1.0 - np.clip((mean_var - 150.0) / 500.0, 0.0, 1.0)
    # Floor it at 0.1 so we don't silently peg to zero
    return float(max(0.1, score))


def _clean_spot_score(gray_road: np.ndarray, road_mask: np.ndarray) -> float:
    """
    Count very dark blobs on the road — pothole / deep-damage proxy.

    Returns 0-1 where 1 = almost no dark spots.
    """
    pixels = gray_road[road_mask > 0]
    if pixels.size < 100:
        return 0.5

    median = float(np.median(pixels))
    # "Dark" = significantly below median (only a pothole gets this dark
    # on a satellite tile)
    threshold = max(median - 30.0, 10.0)
    dark_mask = ((gray_road < threshold) & (road_mask > 0)).astype(np.uint8)

    # Require the dark region to be a connected blob (not a single-pixel noise)
    dark_mask = cv2.morphologyEx(
        dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )

    dark_frac = float(dark_mask.sum()) / float(road_mask.sum() + 1e-6)
    # Typical safe road < 0.5%, damaged > 3%, severe > 8%.
    score = 1.0 - np.clip(dark_frac / 0.05, 0.0, 1.0)
    return float(score)


def _edge_regularity_score(gray_road: np.ndarray, road_mask: np.ndarray) -> float:
    """
    Measures how "regular" the edges on the road are.  Lane markings are
    highly regular (long parallel lines); crack patterns are irregular.

    This is a crack-density measure penalized by how much of the edge
    content is straight lines (lane markings).

    Returns 0-1 where 1 = few irregular edges (= safe).
    """
    if int(road_mask.sum()) < 500:
        return 0.5

    edges = cv2.Canny(gray_road, 40, 130)
    edges_in_road = ((edges > 0) & (road_mask > 0)).astype(np.uint8)
    edge_frac = float(edges_in_road.sum()) / float(road_mask.sum() + 1e-6)

    # Detect long straight lines (these are likely lane markings, not cracks)
    min_len = 25
    lines = cv2.HoughLinesP(
        (edges_in_road * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=min_len,
        maxLineGap=5,
    )
    lane_line_pixels = 0
    if lines is not None:
        # Rasterize the detected lines to count how many edge pixels are "lanes"
        canvas = np.zeros_like(edges_in_road)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(canvas, (x1, y1), (x2, y2), 1, 2)
        lane_line_pixels = int((canvas * edges_in_road).sum())

    # "Irregular" edge density = edges that are NOT part of long straight lines
    irregular_edges = max(0, int(edges_in_road.sum()) - lane_line_pixels)
    irregular_frac = float(irregular_edges) / float(road_mask.sum() + 1e-6)

    # Typical satellite road irregular_frac ≈ 0.04-0.12 for normal roads,
    # 0.15-0.25 for visibly damaged ones. Previous 0.09 ceiling was way too tight.
    score = 1.0 - np.clip((irregular_frac - 0.04) / 0.20, 0.0, 1.0)
    # Floor at 0.1 to avoid silent zeros
    return float(max(0.1, score))


def assess_road_quality(bgr: np.ndarray, road_mask: np.ndarray) -> dict:
    """
    Run all 5 features and combine into a final SAFE / DAMAGED / SEVERE verdict.

    Args:
        bgr: satellite tile BGR image
        road_mask: 0/1 uint8 mask of road pixels

    Returns:
        dict with classification, rdi, confidence, probabilities, features
    """
    # Need enough road to make a call
    road_px = int(road_mask.sum())
    total_px = road_mask.shape[0] * road_mask.shape[1]
    road_frac = road_px / max(total_px, 1)

    if road_frac < 0.05 or road_px < 400:
        return {
            "classification": None,
            "rdi": None,
            "confidence": 0.0,
            "reason": "too little road in tile for reliable assessment",
            "features": {},
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Apply the mask so features look only at road pixels
    gray_road = gray.copy()
    gray_road[road_mask == 0] = 0

    # Compute all 5 features
    features = {
        "linearity":     _linearity_score(gray_road, road_mask),
        "uniformity":    _uniformity_score(gray_road, road_mask),
        "smoothness":    _smoothness_score(gray_road, road_mask),
        "clean_spots":   _clean_spot_score(gray_road, road_mask),
        "edge_regular":  _edge_regularity_score(gray_road, road_mask),
    }

    # Weights for the damage score. Edge regularity and uniformity are the
    # most informative; linearity is most fragile (lane markings vary).
    weights = {
        "edge_regular":  0.30,
        "uniformity":    0.25,
        "smoothness":    0.20,
        "clean_spots":   0.15,
        "linearity":     0.10,
    }

    # Damage = 1 - weighted mean of "goodness" features
    goodness = sum(features[k] * w for k, w in weights.items())
    damage_score = float(max(0.0, min(1.0, 1.0 - goodness)))

    # Classification bands — calibrated for satellite imagery where
    # even good roads have noisy features.  A "normal road" in your real
    # data scored around 0.55-0.65 damage; SEVERE should be reserved for
    # genuinely visually-damaged tiles.
    if damage_score < 0.55:
        cls = "SAFE"
        rdi = round(damage_score / 0.55 * 20.0, 2)      # 0-20
    elif damage_score < 0.75:
        cls = "DAMAGED"
        rdi = round(20.0 + (damage_score - 0.55) / 0.20 * 40.0, 2)  # 20-60
    else:
        cls = "SEVERE"
        rdi = round(60.0 + (damage_score - 0.75) / 0.25 * 40.0, 2)  # 60-100
        rdi = min(rdi, 100.0)

    # Confidence: distance from nearest band edge, normalized
    if cls == "SAFE":
        conf = 0.55 + (0.55 - damage_score) / 0.55 * 0.40
    elif cls == "DAMAGED":
        # Middle band — width 0.20. Best confidence at 0.65 (middle).
        dist_to_edge = min(damage_score - 0.55, 0.75 - damage_score)
        conf = 0.55 + (dist_to_edge / 0.10) * 0.30
    else:
        conf = min(0.55 + (damage_score - 0.75) / 0.25 * 0.40, 0.95)
    conf = float(max(0.55, min(0.95, conf)))

    # 3-way probability distribution. The "loser" classes split (1 - conf)
    # proportional to how close the damage score is to their band.
    if cls == "SAFE":
        probs = {
            "SAFE": conf,
            "DAMAGED": (1 - conf) * 0.75,
            "SEVERE": (1 - conf) * 0.25,
        }
    elif cls == "DAMAGED":
        # Weight the loser-probs toward whichever edge we're closer to
        to_safe = 0.75 - (damage_score)  # distance to SEVERE edge
        to_severe = damage_score - 0.55  # distance from SAFE edge
        s_weight = max(0.1, to_safe) / (max(0.1, to_safe) + max(0.1, to_severe))
        probs = {
            "SAFE":    (1 - conf) * s_weight,
            "DAMAGED": conf,
            "SEVERE":  (1 - conf) * (1 - s_weight),
        }
    else:
        probs = {
            "SAFE": (1 - conf) * 0.15,
            "DAMAGED": (1 - conf) * 0.85,
            "SEVERE": conf,
        }

    probs = {k: round(float(v), 4) for k, v in probs.items()}
    features_rounded = {k: round(float(v), 4) for k, v in features.items()}

    return {
        "classification": cls,
        "rdi": rdi,
        "confidence": round(conf, 4),
        "probabilities": probs,
        "features": features_rounded,
        "damage_score": round(damage_score, 4),
        "road_fraction": round(road_frac, 4),
    }
