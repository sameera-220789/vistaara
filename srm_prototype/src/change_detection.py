#!/usr/bin/env python3
"""
DeepSRM - Change Detection & Disaster Impact Masking Module
----------------------------------------------------------
Performs sub-pixel difference analysis on Super-Resolved pre-event and post-event satellite scenes.
Identifies flood inundation, storm surge submergence, and structural changes.
Lacks calibrated ground sensors, so truthfully labeled as:
'prototype change detection'.
"""

import os
import numpy as np
import cv2
from typing import Dict, Any, Tuple

def run_change_detection(
    pre_bgr: np.ndarray,
    post_bgr: np.ndarray,
    output_path: str,
    threshold: int = 42
) -> Dict[str, Any]:
    """
    Computes absolute radiometric difference between pre and post scenes.
    Generates binary mask and colorized overlay.
    """
    # Resize post to match pre if dimensions differ
    if pre_bgr.shape[:2] != post_bgr.shape[:2]:
        post_bgr = cv2.resize(post_bgr, (pre_bgr.shape[1], pre_bgr.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    h, w = pre_bgr.shape[:2]

    # Convert to grayscale
    gray_pre = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post_bgr, cv2.COLOR_BGR2GRAY)

    # Absolute difference
    diff = cv2.absdiff(gray_pre, gray_post)

    # Thresholding
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Morphological opening and closing to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)

    # Statistics
    total_pixels = h * w
    changed_pixels = int(np.count_nonzero(clean_mask))
    changed_pct = round((changed_pixels / total_pixels) * 100.0, 2)

    # Estimate area assuming nominal 2.5m pixel grid (6.25 m2 per pixel)
    pixel_area_m2 = 6.25
    changed_area_ha = round((changed_pixels * pixel_area_m2) / 10000.0, 2)

    # Create Transparent Color Overlay (Red/Crimson for disaster submergence/change)
    overlay_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # RGBA: Red = 239, 68, 68, Alpha = 175
    overlay_rgba[clean_mask > 0] = [239, 68, 68, 175]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(overlay_rgba, cv2.COLOR_RGBA2BGRA))

    # Also save standalone black/white mask
    mask_path = output_path.replace(".png", "_binary.png")
    cv2.imwrite(mask_path, clean_mask)

    return {
        "overlay_path": output_path,
        "mask_path": mask_path,
        "changed_pixels": changed_pixels,
        "total_pixels": total_pixels,
        "changed_pct": changed_pct,
        "estimated_impacted_ha": changed_area_ha,
        "threshold_used": threshold,
        "scientific_disclaimer": "Prototype change detection based on radiometric differencing. Requires hydrological calibration."
    }
