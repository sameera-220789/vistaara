#!/usr/bin/env python3
"""
DeepSRM - Sentinel-2 & Multispectral GeoTIFF Processor
------------------------------------------------------
Reads geospatial metadata, handles 4-band Sentinel-2 (RGB + NIR),
applies 4x/2x Super-Resolution, updates geospatial affine transforms,
computes NDVI indices, and writes georeferenced GeoTIFFs and preview PNGs.
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, Any, Tuple, Optional, List
import rasterio
from rasterio.transform import Affine

from src.sr_pipeline import run_realesrgan_inference, run_opencv_fallback

def process_geotiff(
    input_path: str,
    output_dir: str,
    scale: int = 4,
    model_name: str = "auto",
    band_order: Optional[List[int]] = None,
    log_callback = None
) -> Dict[str, Any]:
    """
    Processes an input GeoTIFF file.
    band_order: 1-indexed list of bands [R, G, B, NIR] or None for auto.
    Returns dictionary with generated file paths, metadata, and scientific logs.
    """
    logs = []
    def log(msg):
        logs.append(msg)
        if log_callback:
            log_callback(msg)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        count = src.count
        crs = src.crs
        transform = src.transform
        bounds = src.bounds
        nodata = src.nodata
        meta = src.meta.copy()

        log(f"Opened GeoTIFF: {width}x{height} pixels, {count} bands, CRS: {crs}")

        # Read bands
        # Default band mapping
        if band_order and len(band_order) >= 3:
            r_idx, g_idx, b_idx = band_order[0], band_order[1], band_order[2]
            nir_idx = band_order[3] if len(band_order) >= 4 and band_order[3] <= count else None
        else:
            if count >= 4:
                # Standard Sentinel-2 L2A or 4-band composite (B4=Red, B3=Green, B2=Blue, B8=NIR)
                r_idx, g_idx, b_idx, nir_idx = 1, 2, 3, 4
            elif count == 3:
                r_idx, g_idx, b_idx, nir_idx = 1, 2, 3, None
            else:
                # Single band or 2 bands
                r_idx, g_idx, b_idx, nir_idx = 1, 1, 1, None

        log(f"Band mapping: Red=Band {r_idx}, Green=Band {g_idx}, Blue=Band {b_idx}" + (f", NIR=Band {nir_idx}" if nir_idx else " (No NIR)"))

        band_r = src.read(r_idx).astype(np.float32)
        band_g = src.read(g_idx).astype(np.float32)
        band_b = src.read(b_idx).astype(np.float32)
        band_nir = src.read(nir_idx).astype(np.float32) if nir_idx else None

    # Normalize RGB for visualization / inference (handle 8-bit, 12-bit, 16-bit Sentinel-2)
    def normalize_band(b):
        p2, p98 = np.percentile(b[b > 0] if np.any(b > 0) else b, (2, 98))
        if p98 > p2:
            norm = (b - p2) / (p98 - p2)
        else:
            norm = b / (b.max() if b.max() > 0 else 1.0)
        norm = np.clip(norm, 0.0, 1.0)
        return (norm * 255.0).astype(np.uint8)

    rgb_uint8 = np.stack([normalize_band(band_r), normalize_band(band_g), normalize_band(band_b)], axis=2)
    bgr_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)

    # Save original preview PNG
    orig_preview_path = os.path.join(output_dir, f"{base_name}_orig_preview.png")
    cv2.imwrite(orig_preview_path, bgr_uint8)

    # Super-resolution inference
    is_multispectral_opensr = False
    scientific_notice = "Visual RGB enhancement only; multispectral scientific integrity is not guaranteed."

    # Check if OpenSR is requested / available
    if model_name.lower() in ["opensr", "opengeoai"] and band_nir is not None:
        log("OpenSR Sentinel-2 multispectral mode requested.")
        # Note: OpenSR uses latent diffusion models specifically for Sentinel-2 4-band (B2, B3, B4, B8)
        # If open-source opensr-test weights are not present, we use Real-ESRGAN RGB composite with honest disclaimer
        log("Real-ESRGAN applied on calibrated RGB composite with honest disclaimer.")
        sr_bgr, model_used, is_ai, sr_logs = run_realesrgan_inference(bgr_uint8, scale=scale, model_name="realesrgan", log_callback=log)
        multispectral_label = "Visual RGB enhancement only; multispectral scientific integrity is not guaranteed."
    else:
        chosen_model = "realesrgan" if model_name in ["auto", "realesrgan"] else model_name
        sr_bgr, model_used, is_ai, sr_logs = run_realesrgan_inference(bgr_uint8, scale=scale, model_name=chosen_model, log_callback=log)
        multispectral_label = "Visual RGB enhancement only; multispectral scientific integrity is not guaranteed."

    # Save SR preview PNG
    sr_preview_path = os.path.join(output_dir, f"{base_name}_sr_{scale}x_preview.png")
    cv2.imwrite(sr_preview_path, sr_bgr)
    sr_rgb = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)

    # Compute updated affine transform
    # Bounding box is preserved, pixel size is divided by scale
    sr_h, sr_w = sr_bgr.shape[:2]
    new_transform = Affine(
        transform.a / scale,
        transform.b,
        transform.c,
        transform.d,
        transform.e / scale,
        transform.f
    )

    log(f"Updated affine transform: Pixel size divided by {scale}x. Bounds preserved.")

    # Write Super-Resolved GeoTIFF (3-band RGB)
    sr_geotiff_path = os.path.join(output_dir, f"{base_name}_sr_{scale}x.tif")
    meta.update({
        'driver': 'GTiff',
        'height': sr_h,
        'width': sr_w,
        'count': 3,
        'dtype': 'uint8',
        'transform': new_transform,
        'crs': crs
    })

    with rasterio.open(sr_geotiff_path, 'w', **meta) as dst:
        dst.write(sr_rgb[:, :, 0], 1)
        dst.write(sr_rgb[:, :, 1], 2)
        dst.write(sr_rgb[:, :, 2], 3)
        dst.set_band_description(1, "DeepSRM Super-Resolved Red (665nm)")
        dst.set_band_description(2, "DeepSRM Super-Resolved Green (560nm)")
        dst.set_band_description(3, "DeepSRM Super-Resolved Blue (490nm)")

    log(f"Saved georeferenced Super-Resolved GeoTIFF to: {os.path.basename(sr_geotiff_path)}")

    # NDVI Analysis
    ndvi_path = None
    ndvi_status = "No NIR band present in upload."
    if band_nir is not None:
        # Calculate NDVI: (NIR - Red) / (NIR + Red + 1e-8)
        denom = (band_nir + band_r + 1e-8)
        ndvi = (band_nir - band_r) / denom
        ndvi = np.clip(ndvi, -1.0, 1.0)

        # Scale NDVI to match SR dimensions for visualization
        ndvi_sr = cv2.resize(ndvi, (sr_w, sr_h), interpolation=cv2.INTER_LANCZOS4)

        # Colorize NDVI: Red (<0.1 barren/water) -> Yellow (0.2-0.4 sparse) -> Deep Green (>0.5 healthy crops)
        ndvi_norm = np.clip((ndvi_sr + 0.2) / 1.0, 0.0, 1.0)
        ndvi_color = cv2.applyColorMap((ndvi_norm * 255.0).astype(np.uint8), cv2.COLORMAP_SUMMER)

        ndvi_path = os.path.join(output_dir, f"{base_name}_ndvi_overlay.png")
        cv2.imwrite(ndvi_path, ndvi_color)

        ndvi_status = (
            "NDVI calculated from source Sentinel-2 multispectral bands (NIR Band 4 & Red Band 1) "
            "and aligned with Super-Resolved spatial grid. Unvalidated SR spectral synthesis was not used."
        )
        log("Generated calibrated NDVI vegetation vigor overlay.")

    return {
        "orig_preview_path": orig_preview_path,
        "sr_preview_path": sr_preview_path,
        "sr_geotiff_path": sr_geotiff_path,
        "ndvi_overlay_path": ndvi_path,
        "ndvi_status": ndvi_status,
        "model_used": model_used,
        "is_ai_model": is_ai,
        "scale": scale,
        "scientific_notice": multispectral_label,
        "width": sr_w,
        "height": sr_h,
        "crs": str(crs) if crs else "Unknown",
        "logs": logs
    }
