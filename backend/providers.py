"""
Satellite imagery providers — Google Static Maps and Sentinel Hub.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests


GOOGLE_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")
SENTINEL_CLIENT_ID = os.environ.get("SENTINEL_CLIENT_ID", "")
SENTINEL_CLIENT_SECRET = os.environ.get("SENTINEL_CLIENT_SECRET", "")


@dataclass
class Tile:
    bgr: np.ndarray
    lat: float
    lon: float
    zoom: int
    provider: str
    meters_per_pixel: float


# ─── Google Static Maps ─────────────────────────────────────

def _gmaps_mpp(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def fetch_gmaps(lat: float, lon: float, zoom: int = 19,
                size: int = 640) -> Tile:
    if not GOOGLE_KEY:
        raise RuntimeError(
            "GOOGLE_MAPS_API_KEY not set. Add it to your .env file."
        )

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap"
        f"?center={lat},{lon}&zoom={zoom}&size={size}x{size}"
        f"&maptype=satellite&key={GOOGLE_KEY}"
    )
    r = requests.get(url, timeout=15)
    r.raise_for_status()

    arr = np.frombuffer(r.content, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Google Maps returned invalid image")

    return Tile(bgr=bgr, lat=lat, lon=lon, zoom=zoom,
                provider="google", meters_per_pixel=_gmaps_mpp(lat, zoom))


# ─── Sentinel Hub ───────────────────────────────────────────

_sentinel_token_cache = {"token": None, "expires_at": 0.0}


def _sentinel_token() -> str:
    cached = _sentinel_token_cache
    if cached["token"] and time.time() < cached["expires_at"] - 60:
        return cached["token"]

    if not (SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET):
        raise RuntimeError(
            "SENTINEL_CLIENT_ID / SENTINEL_CLIENT_SECRET not set."
        )

    r = requests.post(
        "https://services.sentinel-hub.com/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": SENTINEL_CLIENT_ID,
            "client_secret": SENTINEL_CLIENT_SECRET,
        },
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()
    cached["token"] = data["access_token"]
    cached["expires_at"] = time.time() + data.get("expires_in", 3600)
    return cached["token"]


_EVALSCRIPT_TRUECOLOR = """//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02"] }],
    output: { bands: 3 }
  };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
"""


def fetch_sentinel(lat: float, lon: float,
                   span_m: float = 400.0,
                   size: int = 640,
                   date_from: str = "2024-01-01",
                   date_to: str = "2025-01-01") -> Tile:
    token = _sentinel_token()

    R = 6378137.0
    x = R * math.radians(lon)
    y = R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    half = span_m / 2
    bbox = [x - half, y - half, x + half, y + half]

    body = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/3857"},
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{date_from}T00:00:00Z",
                        "to":   f"{date_to}T23:59:59Z",
                    },
                    "maxCloudCoverage": 20,
                    "mosaickingOrder": "leastCC",
                },
            }],
        },
        "output": {
            "width":  size,
            "height": size,
            "responses": [{"identifier": "default",
                           "format": {"type": "image/png"}}],
        },
        "evalscript": _EVALSCRIPT_TRUECOLOR,
    }

    r = requests.post(
        "https://services.sentinel-hub.com/api/v1/process",
        headers={"Authorization": f"Bearer {token}"},
        json=body,
        timeout=30,
    )
    r.raise_for_status()

    arr = np.frombuffer(r.content, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Sentinel Hub returned invalid image")

    return Tile(
        bgr=bgr, lat=lat, lon=lon,
        zoom=-1, provider="sentinel",
        meters_per_pixel=span_m / size,
    )


# ─── Dispatcher ─────────────────────────────────────────────

def fetch_tile(lat: float, lon: float, provider: str = "auto",
               zoom: int = 19, span_m: float = 300.0,
               size: int = 640) -> Tile:
    if provider == "auto":
        if GOOGLE_KEY:
            provider = "google"
        elif SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET:
            provider = "sentinel"
        else:
            raise RuntimeError(
                "No satellite provider configured. "
                "Set GOOGLE_MAPS_API_KEY or SENTINEL_CLIENT_ID/SECRET in .env."
            )

    if provider == "google":
        return fetch_gmaps(lat, lon, zoom=zoom, size=size)
    if provider == "sentinel":
        return fetch_sentinel(lat, lon, span_m=span_m, size=size)
    raise ValueError(f"Unknown provider: {provider}")


# ─── Grid region sampling ───────────────────────────────────

def grid_points(lat: float, lon: float, half_side_m: float,
                n: int = 3) -> list[tuple[float, float]]:
    dlat = half_side_m / 111_320.0
    dlon = half_side_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.01))

    if n < 2:
        return [(lat, lon)]

    points = []
    for i in range(n):
        for j in range(n):
            fy = (i / (n - 1)) * 2 - 1
            fx = (j / (n - 1)) * 2 - 1
            points.append((lat + fy * dlat, lon + fx * dlon))
    return points
