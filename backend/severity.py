"""
VIT severity classifier — fine-tuned on labeled campus photos.

Features:
  - Auto-detects head architecture from checkpoint shapes
  - Test-time augmentation (TTA) — averages predictions across 8 augmented
    views of the input for a more stable output
  - Feature extraction — exposes encoder features for nearest-neighbor
    retrieval against labeled training photos
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_CLASSES_4 = ["NORMAL", "MINOR", "MODERATE", "SEVERE"]
DEFAULT_CLASSES_3 = ["MINOR", "MODERATE", "SEVERE"]
DEFAULT_CLASSES_2 = ["DAMAGED", "SEVERE"]


def _build_severity_model(num_classes: int, head_spec: dict):
    """Build STCrackNetSeverity with a head matching `head_spec`."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ChannelAttention(nn.Module):
        def __init__(self, channels, reduction=8):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channels, max(channels // reduction, 4)),
                nn.ReLU(),
                nn.Linear(max(channels // reduction, 4), channels),
                nn.Sigmoid(),
            )
        def forward(self, x):
            b, c, _, _ = x.shape
            y = self.pool(x).view(b, c)
            return x * self.fc(y).view(b, c, 1, 1)

    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.block(x)

    modules = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    dims_iter = [256] + head_spec["hidden"] + [num_classes]
    dim_cursor = 0

    for i, kind in head_spec["layer_types"]:
        while len(modules) < i:
            modules.append(nn.Identity())
        if kind == "Dropout":
            modules.append(nn.Dropout(0.3))
        elif kind == "Linear":
            modules.append(nn.Linear(dims_iter[dim_cursor], dims_iter[dim_cursor + 1]))
            dim_cursor += 1
        elif kind == "ReLU":
            modules.append(nn.ReLU(inplace=True))

    cls_head = nn.Sequential(*modules)

    class STCrackNetSeverity(nn.Module):
        def __init__(self):
            super().__init__()
            self.rgb_e1 = ConvBlock(3, 64)
            self.rgb_e2 = ConvBlock(64, 128)
            self.rgb_e3 = ConvBlock(128, 256)
            self.edge_e1 = ConvBlock(1, 32)
            self.edge_e2 = ConvBlock(32, 64)
            self.edge_e3 = ConvBlock(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.ca_rgb = ChannelAttention(256)
            self.ca_edge = ChannelAttention(128)
            self.fusion = nn.Sequential(
                nn.Conv2d(384, 256, 1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            )
            self.cls_head = cls_head

        def _sobel(self, x):
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            kx = torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            ky = torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]],
                              dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
            return torch.sqrt(
                F.conv2d(gray, kx, padding=1) ** 2 +
                F.conv2d(gray, ky, padding=1) ** 2 + 1e-6
            )

        def _encode(self, x):
            """Return the 256-channel fused feature map (before classification head)."""
            edge = self._sobel(x)
            r1 = self.rgb_e1(x);             e1 = self.edge_e1(edge)
            r2 = self.rgb_e2(self.pool(r1)); e2 = self.edge_e2(self.pool(e1))
            r3 = self.rgb_e3(self.pool(r2)); e3 = self.edge_e3(self.pool(e2))
            return self.fusion(torch.cat([self.ca_rgb(r3), self.ca_edge(e3)], dim=1))

        def forward(self, x):
            return self.cls_head(self._encode(x))

        def extract_features(self, x):
            """Global-avg-pooled 256-D feature vector for similarity search."""
            fmap = self._encode(x)
            return torch.nn.functional.adaptive_avg_pool2d(fmap, 1).flatten(1)

    return STCrackNetSeverity()


def _infer_head_spec(state_dict):
    """Read cls_head.* keys from the checkpoint and reconstruct the layer layout."""
    linear_items = []
    max_idx = -1
    for k in state_dict.keys():
        if k.startswith("cls_head.") and k.endswith(".weight"):
            w = state_dict[k]
            if w.dim() == 2:
                idx = int(k.split(".")[1])
                linear_items.append((idx, w.shape))
                max_idx = max(max_idx, idx)

    linear_items.sort()
    if not linear_items:
        raise ValueError("No Linear layers found in cls_head")

    dims = [linear_items[0][1][1]]
    for _, shape in linear_items:
        dims.append(shape[0])
    num_classes = dims[-1]
    hidden = dims[1:-1]

    layer_types = []
    for idx in range(2, max_idx + 1):
        if any(li[0] == idx for li in linear_items):
            layer_types.append((idx, "Linear"))
        else:
            next_linear_idx = next((li[0] for li in linear_items if li[0] > idx), None)
            prev_linear_idx = max(
                (li[0] for li in linear_items if li[0] < idx), default=-1
            )
            if next_linear_idx is not None and idx == next_linear_idx - 1:
                layer_types.append((idx, "Dropout"))
            elif prev_linear_idx >= 0 and idx == prev_linear_idx + 1:
                layer_types.append((idx, "ReLU"))
            else:
                layer_types.append((idx, "Dropout"))

    return {"hidden": hidden, "num_classes": num_classes, "layer_types": layer_types}


def load_severity_model(weights_path: str, device: str = "cpu"):
    """Load fine-tuned severity model. Returns (model, class_names) or (None, None)."""
    if not Path(weights_path).exists():
        return None, None
    try:
        import torch
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get("model_state", ckpt)

        spec = _infer_head_spec(state)
        num_classes = spec["num_classes"]

        class_names = ckpt.get("class_names")
        if class_names is None or len(class_names) != num_classes:
            class_names = (DEFAULT_CLASSES_2 if num_classes == 2 else
                           DEFAULT_CLASSES_3 if num_classes == 3 else
                           DEFAULT_CLASSES_4)

        model = _build_severity_model(num_classes, spec).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] severity model: {len(missing)} missing keys")
        if unexpected:
            print(f"[WARN] severity model: {len(unexpected)} unexpected keys")
        model.eval()
        return model, class_names
    except Exception as e:
        print(f"[WARN] severity model load failed: {e}")
        import traceback; traceback.print_exc()
        return None, None


def _preprocess(bgr: np.ndarray) -> np.ndarray:
    """Standard preprocessing: crop bottom 60%, resize to 256×256, RGB."""
    h = bgr.shape[0]
    roi = bgr[int(h * 0.4):, :, :]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (256, 256))


def predict_severity(model, class_names, bgr: np.ndarray, device: str,
                     use_tta: bool = True) -> dict:
    """
    Predict severity. With use_tta=True, averages predictions across 8
    augmented views (horizontal flip + brightness jitter + small rotation
    variants) for more stable output.
    """
    import torch

    base_rgb = _preprocess(bgr)

    # Build TTA variants
    views = [base_rgb]
    if use_tta:
        # Horizontal flip
        views.append(base_rgb[:, ::-1, :].copy())

        # Brightness variants (darker and brighter)
        for alpha, beta in [(0.85, -10), (1.15, 10), (0.95, 5), (1.05, -5)]:
            v = np.clip(base_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            views.append(v)

        # Small rotations
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((128, 128), angle, 1.0)
            v = cv2.warpAffine(base_rgb, M, (256, 256), borderMode=cv2.BORDER_REFLECT_101)
            views.append(v)

    # Stack into a batch
    batch = np.stack(views, axis=0).astype(np.float32) / 255.0
    batch = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs_all = torch.softmax(logits, dim=1).cpu().numpy()

    # Average probabilities across all views
    probs = probs_all.mean(axis=0)
    pred_idx = int(probs.argmax())
    n = len(class_names)

    if n == 2:
        score_map = [40, 85]
    elif n == 3:
        score_map = [20, 55, 90]
    else:
        score_map = [0, 33, 66, 100]

    rdi = float(round(sum(float(probs[i]) * score_map[i] for i in range(n)), 2))

    # Standard deviation across TTA views — quantifies uncertainty
    tta_std = float(round(probs_all.std(axis=0).max(), 4)) if use_tta else 0.0

    return {
        "class_name": class_names[pred_idx],
        "class_id": pred_idx,
        "probabilities": {
            class_names[i]: float(round(float(probs[i]), 4))
            for i in range(n)
        },
        "confidence": float(round(float(probs[pred_idx]), 4)),
        "rdi": rdi,
        "tta_views": len(views),
        "tta_disagreement": tta_std,
    }


def extract_features(model, bgr: np.ndarray, device: str) -> np.ndarray:
    """
    Extract a 256-D feature vector from a photo using the severity model's encoder.
    Used for nearest-neighbor similarity search across labeled VIT photos.
    """
    import torch

    rgb = _preprocess(bgr).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.extract_features(tensor).squeeze().cpu().numpy()

    # L2-normalize so cosine similarity = dot product
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def retrieve_similar(query_feat: np.ndarray, catalog_feats: dict,
                     top_k: int = 3) -> list:
    """
    Find the top-k most similar photos in a labeled catalog.
    `catalog_feats` maps photo_id → L2-normalized feature vector.
    Returns list of dicts with photo_id and cosine similarity.
    """
    results = []
    for photo_id, feat in catalog_feats.items():
        sim = float(np.dot(query_feat, feat))
        results.append({"photo_id": photo_id, "similarity": round(sim, 4)})
    results.sort(key=lambda x: -x["similarity"])
    return results[:top_k]
