#!/usr/bin/env python3
"""
DeepSRM - Heuristic Confidence & Spatial Uncertainty Overlay
------------------------------------------------------------
Computes local spatial edge gradient confidence from the super-resolved output.
Provides visual confidence indicator to guide GIS operators and surveyors:
- Green (>= 80%): High Confidence (sharp boundary gradients, clear bunds/roads)
- Amber/Yellow (50 - 80%): Medium Confidence (subtle textures)
- Red (< 50%): Ambiguous / Low Confidence (flat, occluded, or low-contrast zones)
"""

import os
import numpy as np
import cv2
from typing import Tuple

def generate_confidence_overlay(sr_bgr: np.ndarray, output_path: str) -> Tuple[str, dict]:
    """
    Generates a 4-channel RGBA confidence overlay image.
    """
    gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Multi-scale edge gradients
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)

    # Normalize magnitude
    norm_mag = cv2.normalize(mag, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    # Blur to create regional confidence heat-zones
    blurred = cv2.GaussianBlur(norm_mag, (15, 15), 3.0)

    # Mix base confidence (0.45) with edge fidelity boost (up to 0.55)
    conf = np.clip(0.45 + 0.55 * (blurred * 1.6), 0.0, 1.0)

    h, w = sr_bgr.shape[:2]
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Vectorized color mapping
    # Red for low (<0.55), Amber for mid (0.55-0.75), Green for high (>=0.75)
    high_mask = (conf >= 0.75)
    mid_mask = (conf >= 0.55) & (~high_mask)
    low_mask = (~high_mask) & (~mid_mask)

    # High -> Emerald Green (RGB: 16, 185, 129, Alpha: 130)
    overlay_rgba[high_mask] = [16, 185, 129, 130]

    # Mid -> Amber (RGB: 245, 158, 11, Alpha: 120)
    overlay_rgba[mid_mask] = [245, 158, 11, 120]

    # Low -> Red (RGB: 239, 68, 68, Alpha: 110)
    overlay_rgba[low_mask] = [239, 68, 68, 110]

    # Smooth the alpha transitions slightly
    alpha_ch = cv2.GaussianBlur(overlay_rgba[:, :, 3], (7, 7), 1.5)
    overlay_rgba[:, :, 3] = alpha_ch

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA))

    stats = {
        "high_confidence_pct": round(float(np.count_nonzero(high_mask) / conf.size * 100), 1),
        "mid_confidence_pct": round(float(np.count_nonzero(mid_mask) / conf.size * 100), 1),
        "low_confidence_pct": round(float(np.count_nonzero(low_mask) / conf.size * 100), 1),
        "mean_confidence_score": round(float(np.mean(conf) * 100), 1)
    }

    return output_path, stats
