"""
Core analysis pipeline.

Two modes:
  - 'pavement': close-up road photo → STCrackNet directly.
  - 'satellite': lat/lon → fetch tile → extract road → crack heuristic → metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# Road extraction (satellite)
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Land-cover classification (satellite)
# ─────────────────────────────────────────────────────────────
# Classifies each pixel as one of:
#   0 = unknown / other
#   1 = road (asphalt)
#   2 = building (rooftop)
#   3 = vegetation (trees, grass, crops)
#   4 = bare ground (soil, dirt)
# ─────────────────────────────────────────────────────────────

CLASS_UNKNOWN   = 0
CLASS_ROAD      = 1
CLASS_BUILDING  = 2
CLASS_VEGETATION = 3
CLASS_BARE      = 4

CLASS_NAMES = {
    CLASS_UNKNOWN:   "unknown",
    CLASS_ROAD:      "road",
    CLASS_BUILDING:  "building",
    CLASS_VEGETATION: "vegetation",
    CLASS_BARE:      "bare ground",
}

# BGR colours for visualising the land-cover map
CLASS_COLORS = {
    CLASS_UNKNOWN:   (80, 80, 80),
    CLASS_ROAD:      (200, 200, 50),   # cyan-ish
    CLASS_BUILDING:  (60, 60, 220),    # red (rooftops often reddish/tan)
    CLASS_VEGETATION: (80, 180, 80),   # green
    CLASS_BARE:      (80, 140, 200),   # tan/orange
}


def classify_land_cover(bgr: np.ndarray) -> np.ndarray:
    """
    Return an H×W uint8 array where each pixel is a CLASS_* value.
    Uses colour (HSV), greenness index, and local texture.
    """
    # Colour cues
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    b, g, r = cv2.split(bgr)

    # Vegetation index: excess-green (VEG = 2G - R - B)
    veg_index = (2.0 * g.astype(np.int16) -
                 r.astype(np.int16) -
                 b.astype(np.int16))

    # Redness: dominance of red over blue (rooftops, soil are warm)
    redness = r.astype(np.int16) - b.astype(np.int16)

    # Local texture via standard deviation on a 9x9 window
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    mean_sq = cv2.blur(gray * gray, (9, 9))
    texture = np.sqrt(np.maximum(mean_sq - mean * mean, 0))

    out = np.zeros(bgr.shape[:2], dtype=np.uint8)

    # 1) Vegetation — strong green dominance
    veg_mask = (veg_index > 12) & (g > 55)
    out[veg_mask] = CLASS_VEGETATION

    # 2) Warm-coloured buildings/bare — red dominates blue
    #    Distinguish building (bright, textured) from bare (moderate, uniform)
    warm_mask = (redness > 18) & (veg_index < 10) & (~veg_mask)

    building_warm = warm_mask & (v > 110) & (texture > 10)
    bare_mask     = warm_mask & (~building_warm)
    out[building_warm & (out == 0)] = CLASS_BUILDING
    out[bare_mask     & (out == 0)] = CLASS_BARE

    # 3) Bright, textured, neutral-colour patches → buildings
    #    (many rooftops are grey/white, not red)
    bright_building = (
        (v > 165) &
        (s < 70) &
        (texture > 14) &
        (out == 0)
    )
    out[bright_building] = CLASS_BUILDING

    # 4) Road — grey, low saturation, mid value, SMOOTH texture
    road_mask = (
        (s < 55) &
        (v >= 45) & (v < 165) &
        (texture < 18) &
        (out == 0)
    )
    out[road_mask] = CLASS_ROAD

    # 5) Remaining textured-grey → building (e.g. complex rooftops)
    rest_building = (
        (s < 70) &
        (texture >= 18) &
        (out == 0)
    )
    out[rest_building] = CLASS_BUILDING

    # Clean up road mask: real roads are large elongated components
    road_only = (out == CLASS_ROAD).astype(np.uint8)
    road_only = cv2.morphologyEx(road_only, cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    road_only = cv2.morphologyEx(road_only, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(road_only, 8)
    min_road_area = (bgr.shape[0] * bgr.shape[1]) * 0.003
    cleaned_road = np.zeros_like(road_only)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_road_area:
            cleaned_road[labels == i] = 1

    # Any "road" that didn't survive filtering → building
    was_road_removed = (out == CLASS_ROAD) & (cleaned_road == 0)
    out[was_road_removed] = CLASS_BUILDING
    out[cleaned_road > 0] = CLASS_ROAD

    return out


def land_cover_summary(cover: np.ndarray) -> dict:
    """Percentages of each land-cover class."""
    total = float(cover.size)
    return {
        CLASS_NAMES[c]: round(float((cover == c).sum()) / total, 4)
        for c in (CLASS_ROAD, CLASS_BUILDING, CLASS_VEGETATION, CLASS_BARE, CLASS_UNKNOWN)
    }


def dominant_class(cover: np.ndarray, cx: int, cy: int,
                   window: int = 40) -> int:
    """Most common class in a window around (cx, cy)."""
    h, w = cover.shape
    x0 = max(0, cx - window); x1 = min(w, cx + window)
    y0 = max(0, cy - window); y1 = min(h, cy + window)
    patch = cover[y0:y1, x0:x1]
    if patch.size == 0:
        return CLASS_UNKNOWN
    counts = np.bincount(patch.ravel(), minlength=5)
    return int(counts.argmax())


def render_land_cover(cover: np.ndarray, base: np.ndarray = None) -> np.ndarray:
    """Render the land-cover map as a coloured BGR image blended with the source."""
    h, w = cover.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c, col in CLASS_COLORS.items():
        vis[cover == c] = col

    if base is not None:
        out = (base.astype(np.float32) * 0.55 +
               vis.astype(np.float32) * 0.45)
        return np.clip(out, 0, 255).astype(np.uint8)
    return vis


# ─────────────────────────────────────────────────────────────
# Legacy helpers kept for backwards compatibility
# ─────────────────────────────────────────────────────────────

def extract_road_mask(bgr: np.ndarray) -> np.ndarray:
    """Legacy: returns a 0/1 mask of the road class from land-cover."""
    cover = classify_land_cover(bgr)
    return (cover == CLASS_ROAD).astype(np.uint8)


def detect_cracks_on_road(bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """
    Detect crack-like features only within the road mask.
    Uses multi-scale black-hat + Canny + elongation filter.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    bh_small = cv2.morphologyEx(
        blur, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    bh_large = cv2.morphologyEx(
        blur, cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
    )
    enhanced = cv2.max(bh_small, bh_large)

    edges = cv2.Canny(enhanced, 20, 60)
    masked = (edges > 0).astype(np.uint8) * (road_mask > 0).astype(np.uint8)

    connected = cv2.dilate(
        masked, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), 1,
    )

    # Keep only elongated components
    num, labels, stats, _ = cv2.connectedComponentsWithStats(connected, 8)
    out = np.zeros_like(connected)
    for i in range(1, num):
        _, _, w, h, area = stats[i]
        if area < 8:
            continue
        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        aspect = long_side / short_side
        if aspect >= 2.5 and area <= 0.02 * connected.size:
            out[labels == i] = 1

    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

@dataclass
class DamageMetrics:
    crack_coverage: float
    crack_density: float
    mean_width_px: float
    connectivity: float
    severity_score: float
    rdi: float
    classification: str
    road_coverage: float

    def as_dict(self) -> dict:
        return asdict(self)


def _crack_width(crack_mask: np.ndarray) -> float:
    if crack_mask.sum() == 0:
        return 0.0
    dist = cv2.distanceTransform(
        crack_mask.astype(np.uint8) * 255, cv2.DIST_L2, 5
    )
    nonzero = dist[crack_mask > 0]
    if nonzero.size == 0:
        return 0.0
    return float(nonzero.mean() * 2)


def _connectivity(crack_mask: np.ndarray) -> float:
    if crack_mask.sum() == 0:
        return 0.0
    num_cc, _, stats, _ = cv2.connectedComponentsWithStats(
        crack_mask.astype(np.uint8), 8
    )
    if num_cc <= 1:
        return 0.0
    total = crack_mask.sum()
    largest = stats[1:, cv2.CC_STAT_AREA].max() if num_cc > 1 else 0
    return float(largest / (total + 1e-6))


def compute_metrics(crack_mask: np.ndarray,
                    road_mask: Optional[np.ndarray] = None) -> DamageMetrics:
    total_px = crack_mask.size
    crack_px = float(crack_mask.sum())

    if road_mask is not None:
        road_px = float(road_mask.sum()) + 1e-6
        coverage = crack_px / road_px
        road_frac = road_px / total_px
    else:
        coverage = crack_px / total_px
        road_frac = 1.0

    density = crack_px / total_px
    width = _crack_width(crack_mask)
    conn = _connectivity(crack_mask)

    # Severity formula — calibrated for cropped pavement regions.
    # Coverage is now measured over the road crop (not the whole image),
    # so even small absolute percentages indicate real damage.
    severity = float(np.clip(
        0.55 * min(coverage * 12, 1.0) +   # more sensitive to coverage
        0.25 * min(width / 4.0, 1.0) +     # widths over ~4px are concerning
        0.20 * conn,
        0.0, 1.0,
    ))

    rdi = round(severity * 100, 2)

    # Adjusted thresholds for VIT-style wide-angle photos
    if severity < 0.12:
        cls = "NORMAL"
    elif severity < 0.30:
        cls = "MINOR"
    elif severity < 0.55:
        cls = "MODERATE"
    else:
        cls = "SEVERE"

    return DamageMetrics(
        crack_coverage=round(coverage, 4),
        crack_density=round(density, 4),
        mean_width_px=round(width, 2),
        connectivity=round(conn, 4),
        severity_score=round(severity, 4),
        rdi=rdi,
        classification=cls,
        road_coverage=round(road_frac, 4),
    )


# ─────────────────────────────────────────────────────────────
# STCrackNet inference (pavement mode)
# ─────────────────────────────────────────────────────────────

def stcracknet_infer(model, bgr: np.ndarray, device: str,
                     threshold: float = 0.5) -> np.ndarray:
    """Run STCrackNet on a BGR image → binary crack mask at input resolution."""
    import torch  # lazy — satellite-only deployments don't need torch

    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256))
    tensor = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    mask_small = (probs > threshold).astype(np.uint8)
    mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────

def make_overlay(bgr: np.ndarray, crack_mask: np.ndarray,
                 road_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Compose annotated overlay: dim non-road, highlight cracks in red."""
    out = bgr.copy().astype(np.float32)

    if road_mask is not None:
        non_road = (road_mask == 0)
        out[non_road] = out[non_road] * 0.45

    crack = crack_mask > 0
    glow = cv2.dilate(crack.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
    glow_only = (glow > 0) & (~crack)

    out[glow_only] = out[glow_only] * 0.6 + np.array([0, 0, 180]) * 0.4
    out[crack] = np.array([40, 60, 255])

    return np.clip(out, 0, 255).astype(np.uint8)


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# ─────────────────────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────────────────────

def crop_to_pavement(bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Detect the largest pavement (grey/low-saturation) region in the lower
    portion of the image and crop to it. Returns (cropped_bgr, (x, y, w, h)).

    For street-view photos where sky/trees/buildings dominate, this keeps
    only the actual road surface so STCrackNet gets the right input.
    """
    h, w = bgr.shape[:2]

    # Focus the search on the bottom 60% of the image (roads are almost
    # always in the foreground of a street-view photo)
    roi_top = int(h * 0.40)
    roi = bgr[roi_top:, :, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]

    # Pavement: low saturation (grey), mid value (not pitch black, not white)
    pavement = ((s < 60) & (v > 40) & (v < 200)).astype(np.uint8)

    # Clean up with morphology — close gaps from lane markings and shadows
    pavement = cv2.morphologyEx(
        pavement, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)),
    )
    pavement = cv2.morphologyEx(
        pavement, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )

    # Find the biggest connected component → the road
    num, labels, stats, _ = cv2.connectedComponentsWithStats(pavement, 8)
    if num <= 1:
        return bgr, (0, 0, w, h)

    # Largest (ignoring background label 0)
    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, cw, ch, area = stats[best_idx]

    # Reject if the "road" is too tiny (<5% of ROI) — fall back to full image
    if area < 0.05 * roi.shape[0] * roi.shape[1]:
        return bgr, (0, 0, w, h)

    # Translate back to full-image coords
    y_full = y + roi_top

    # Slight padding
    pad = 8
    x0 = max(0, x - pad)
    y0 = max(0, y_full - pad)
    x1 = min(w, x + cw + pad)
    y1 = min(h, y_full + ch + pad)

    cropped = bgr[y0:y1, x0:x1, :]
    return cropped, (x0, y0, x1 - x0, y1 - y0)


def stcracknet_tiled(model, bgr: np.ndarray, device: str,
                     tile: int = 256, overlap: int = 32,
                     threshold: float = 0.5) -> np.ndarray:
    """
    Run STCrackNet over an image in 256×256 tiles and stitch the masks.
    This preserves crack detail that would be lost if we resized a big
    image down to 256×256 in one shot.
    """
    import torch
    h, w = bgr.shape[:2]

    # If image is already small-ish, one-shot inference is fine
    if max(h, w) <= tile * 1.5:
        return stcracknet_infer(model, bgr, device, threshold)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask_acc = np.zeros((h, w), dtype=np.float32)
    count_acc = np.zeros((h, w), dtype=np.float32)

    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    # Ensure we cover the right/bottom edge
    if ys and ys[-1] != h - tile:
        ys.append(max(h - tile, 0))
    if xs and xs[-1] != w - tile:
        xs.append(max(w - tile, 0))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    for y in ys:
        for x in xs:
            patch = rgb[y:y + tile, x:x + tile]
            ph, pw = patch.shape[:2]
            # Pad if smaller than tile (edge case)
            if ph != tile or pw != tile:
                padded = np.zeros((tile, tile, 3), dtype=patch.dtype)
                padded[:ph, :pw] = patch
                patch = padded

            t = torch.from_numpy(patch).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            t = t.to(device)
            with torch.no_grad():
                probs = torch.sigmoid(model(t)).squeeze().cpu().numpy()

            mask_acc[y:y + ph, x:x + pw] += probs[:ph, :pw]
            count_acc[y:y + ph, x:x + pw] += 1.0

    avg = mask_acc / np.maximum(count_acc, 1e-6)
    return (avg > threshold).astype(np.uint8)


def analyze_pavement(model, bgr: np.ndarray, device: str) -> dict:
    """
    Pavement-mode:
      1. Crop to the road surface (for wide-angle street photos).
      2. Run STCrackNet in 256×256 tiles to preserve crack detail.
      3. Compute metrics on the cropped region.
    """
    # Step 1: crop to road
    cropped, (cx, cy, cw, ch) = crop_to_pavement(bgr)

    # Step 2: tiled inference on the crop
    crack_mask_crop = stcracknet_tiled(model, cropped, device)

    # Step 3: metrics over the crop (denominator = road, not whole image)
    road_mask_crop = np.ones(crack_mask_crop.shape, dtype=np.uint8)
    metrics = compute_metrics(crack_mask_crop, road_mask_crop)

    # Build a full-size crack mask for display (paste the cropped mask
    # back into a full-size canvas)
    full_h, full_w = bgr.shape[:2]
    crack_mask_full = np.zeros((full_h, full_w), dtype=np.uint8)
    crack_mask_full[cy:cy + ch, cx:cx + cw] = crack_mask_crop

    overlay = make_overlay(bgr, crack_mask_full, None)
    # Also draw the crop box on the overlay so the user sees what the model saw
    cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), (0, 229, 255), 2)

    return {
        "metrics": metrics.as_dict(),
        "input_png": encode_png(bgr),
        "mask_png": encode_png(crack_mask_full * 255),
        "overlay_png": encode_png(overlay),
        "mode": "pavement",
        "model": "STCrackNet (cropped + tiled)",
    }


def analyze_satellite(bgr: np.ndarray) -> dict:
    """
    Satellite-mode: classify land cover (road/building/vegetation/bare),
    report what's under the tile centre, return visualisations.

    Does NOT make damage claims — crack detection from ~0.3 m/pixel
    satellite tiles is not reliable with classical CV, and STCrackNet
    was trained on close-up pavement not overhead imagery.
    """
    h, w = bgr.shape[:2]

    # Land-cover classification
    cover = classify_land_cover(bgr)
    cover_summary = land_cover_summary(cover)

    # What is under the tile centre?
    centre_class = dominant_class(cover, w // 2, h // 2, window=min(w, h) // 10)
    centre_label = CLASS_NAMES[centre_class]

    # Percentages
    total_px = float(cover.size)
    road_frac = float((cover == CLASS_ROAD).sum()) / total_px
    bld_frac  = float((cover == CLASS_BUILDING).sum()) / total_px
    veg_frac  = float((cover == CLASS_VEGETATION).sum()) / total_px
    bare_frac = float((cover == CLASS_BARE).sum()) / total_px

    # Visualisations
    land_vis = render_land_cover(cover, base=bgr)

    return {
        "metrics": {
            "dominant_cover": centre_label,
            "cover_breakdown": cover_summary,
            "road_coverage": round(road_frac, 4),
            "building_coverage": round(bld_frac, 4),
            "vegetation_coverage": round(veg_frac, 4),
            "bare_coverage": round(bare_frac, 4),
            "classification": centre_label.upper().replace(" ", "_"),
        },
        "input_png": encode_png(bgr),
        "land_png": encode_png(land_vis),
        "mode": "satellite",
        "model": "Land Cover Viewer",
    }