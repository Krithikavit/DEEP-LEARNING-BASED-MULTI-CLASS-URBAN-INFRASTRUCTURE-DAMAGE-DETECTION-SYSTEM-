"""
RoadSense · RDD YOLO Stage 1 — test the pretrained model before integrating.

This script:
  1. Installs ultralytics if not already there
  2. Downloads oracl4's YOLOv8 model (trained on RDD2022 Japan+India)
  3. Runs it on one VIT photo of your choosing
  4. Saves an annotated output image showing detected damage

Run from your roadsense folder:
  python test_rdd_yolo.py

If this succeeds, we proceed to Stage 2 (integrate into backend).
If it fails, we pivot. Either way, zero impact on your existing project.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
RDD_WEIGHTS = WEIGHTS_DIR / "YOLOv8_Small_RDD.pt"
TEST_PHOTO = PROJECT_ROOT / "data" / "vit_photos" / "road8.jpg"  # visibly damaged
OUTPUT_DIR = PROJECT_ROOT / "rdd_test_output"

# The oracl4 model is hosted as a LFS file on GitHub. The raw download URL:
RDD_MODEL_URL = (
    "https://github.com/oracl4/RoadDamageDetection/"
    "raw/main/models/YOLOv8_Small_RDD.pt"
)


def step(msg):
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print("=" * 60)


def pip_install(package: str):
    print(f"  pip install {package} ...")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", package],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:500]}")
        return False
    print(f"  ✓ installed")
    return True


def main():
    step("STAGE 1 — Verify RDD YOLO works on your machine")

    # 1. Check ultralytics
    print("\n[1/4] Checking for ultralytics library...")
    try:
        import ultralytics  # type: ignore
        print(f"  ✓ ultralytics already installed (version {ultralytics.__version__})")
    except ImportError:
        print("  ultralytics not found. Installing...")
        if not pip_install("ultralytics"):
            print("\n  ERROR: Could not install ultralytics.")
            print("  Try manually: pip install ultralytics")
            return 1
        import ultralytics  # noqa
        print(f"  ✓ ultralytics installed (version {ultralytics.__version__})")

    # 2. Download model weights if missing
    print("\n[2/4] Getting RDD YOLO weights...")
    WEIGHTS_DIR.mkdir(exist_ok=True)
    if RDD_WEIGHTS.exists() and RDD_WEIGHTS.stat().st_size > 1_000_000:
        size_mb = RDD_WEIGHTS.stat().st_size / 1024 / 1024
        print(f"  ✓ already downloaded ({size_mb:.1f} MB) at {RDD_WEIGHTS}")
    else:
        print(f"  downloading from {RDD_MODEL_URL}")
        print(f"  → {RDD_WEIGHTS}")
        print("  (this is about 22 MB, may take a minute)")
        try:
            urlretrieve(RDD_MODEL_URL, RDD_WEIGHTS)
            size_mb = RDD_WEIGHTS.stat().st_size / 1024 / 1024
            print(f"  ✓ downloaded ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ERROR: download failed — {e}")
            print()
            print("  MANUAL FALLBACK:")
            print("    1. Open https://github.com/oracl4/RoadDamageDetection in a browser")
            print("    2. Click the 'models' folder")
            print("    3. Click 'YOLOv8_Small_RDD.pt' → click 'Download raw file' icon")
            print(f"    4. Save it to: {RDD_WEIGHTS}")
            print("    5. Re-run this script")
            return 1

    # 3. Load the model
    print("\n[3/4] Loading YOLO model...")
    try:
        from ultralytics import YOLO
        model = YOLO(str(RDD_WEIGHTS))
        print(f"  ✓ loaded model")
        print(f"  classes: {model.names}")
    except Exception as e:
        print(f"  ERROR: could not load — {e}")
        return 1

    # 4. Run on a VIT photo
    print(f"\n[4/4] Running detection on {TEST_PHOTO.name}...")
    if not TEST_PHOTO.exists():
        # Fall back to any jpg in vit_photos
        alts = list((PROJECT_ROOT / "data" / "vit_photos").glob("*.jpg"))
        if not alts:
            print(f"  ERROR: no test photos found in data/vit_photos/")
            return 1
        test_photo = alts[0]
        print(f"  road8 not found, using {test_photo.name}")
    else:
        test_photo = TEST_PHOTO

    OUTPUT_DIR.mkdir(exist_ok=True)
    try:
        results = model(str(test_photo), conf=0.25, save=True, project=str(OUTPUT_DIR),
                        name="run", exist_ok=True)
        r = results[0]
        n_detections = len(r.boxes) if r.boxes is not None else 0
        print(f"  ✓ inference complete — {n_detections} detections")

        if n_detections > 0:
            print("\n  Detected damage:")
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"    [{i + 1}] {cls_name:20s}  conf={conf:.2f}  "
                      f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")

        # Find the saved annotated image
        out_files = list((OUTPUT_DIR / "run").glob("*.jpg")) + \
                    list((OUTPUT_DIR / "run").glob("*.png"))
        if out_files:
            print(f"\n  Annotated image saved to:")
            print(f"    {out_files[0]}")
            print(f"\n  Open it to see the bounding boxes the model drew.")
    except Exception as e:
        print(f"  ERROR: inference failed — {e}")
        import traceback; traceback.print_exc()
        return 1

    step("STAGE 1 COMPLETE")
    print("\nIf the annotated image shows real bounding boxes on damage,")
    print("we're good to proceed to Stage 2 (integrate into the dashboard).")
    print("\nPaste back:")
    print("  1. The detection list printed above")
    print("  2. Open the annotated image and describe what you see")
    print("     (boxes on real damage? wrong places? nothing?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
