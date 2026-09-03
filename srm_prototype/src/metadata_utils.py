#!/usr/bin/env python3
"""
DeepSRM - Geospatial Metadata & Image Inspection Utilities
----------------------------------------------------------
Inspects uploaded satellite imagery (PNG, JPG, JPEG, TIF, TIFF).
Reads geospatial tags, CRS, bounds, affine transform, pixel ground resolution.
"""

import os
from typing import Dict, Any, Optional

def inspect_image_file(file_path: str) -> Dict[str, Any]:
    """
    Analyzes an uploaded file and returns rich metadata.
    """
    ext = os.path.splitext(file_path)[1].lower()
    is_geotiff = ext in ['.tif', '.tiff']

    metadata: Dict[str, Any] = {
        "filename": os.path.basename(file_path),
        "extension": ext,
        "is_geotiff": is_geotiff,
        "format_type": "GeoTIFF" if is_geotiff else ("PNG" if ext == ".png" else "JPEG"),
        "file_size_bytes": os.path.getsize(file_path),
        "width": 0,
        "height": 0,
        "bands": 3,
        "crs": None,
        "crs_epsg": None,
        "pixel_resolution_m": None,
        "pixel_size_x": None,
        "pixel_size_y": None,
        "bounds": None,
        "affine_transform": None,
        "nodata": None,
        "band_descriptions": [],
        "has_nir": False,
        "resolution_label_template": "4x pixel upscaling applied; physical ground resolution requires validation."
    }

    if is_geotiff:
        try:
            import rasterio
            with rasterio.open(file_path) as src:
                metadata["width"] = src.width
                metadata["height"] = src.height
                metadata["bands"] = src.count
                metadata["nodata"] = src.nodata

                if src.crs:
                    metadata["crs"] = str(src.crs)
                    if src.crs.to_epsg():
                        metadata["crs_epsg"] = src.crs.to_epsg()

                if src.transform:
                    metadata["affine_transform"] = list(src.transform)[:6]
                    # Pixel size
                    res_x = abs(src.transform.a)
                    res_y = abs(src.transform.e)
                    metadata["pixel_size_x"] = res_x
                    metadata["pixel_size_y"] = res_y

                    # Determine physical resolution
                    # In projected coordinate systems (like UTM), units are meters
                    if src.crs and (src.crs.is_projected or "UTM" in str(src.crs) or res_x > 0.01):
                        metadata["pixel_resolution_m"] = round(float(res_x), 2)
                    elif res_x <= 0.001: # Likely WGS84 degrees (~111km per deg at equator)
                        approx_m = res_x * 111320.0
                        metadata["pixel_resolution_m"] = round(float(approx_m), 2)

                if src.bounds:
                    metadata["bounds"] = {
                        "left": float(src.bounds.left),
                        "bottom": float(src.bounds.bottom),
                        "right": float(src.bounds.right),
                        "top": float(src.bounds.top)
                    }

                # Check if band count >= 4 (RGB + NIR)
                metadata["has_nir"] = src.count >= 4
                for i in range(1, src.count + 1):
                    desc = src.descriptions[i - 1] if src.descriptions and len(src.descriptions) >= i else None
                    if not desc:
                        if i == 1: desc = "Band 1 (Red / B04)"
                        elif i == 2: desc = "Band 2 (Green / B03)"
                        elif i == 3: desc = "Band 3 (Blue / B02)"
                        elif i == 4: desc = "Band 4 (Near Infrared / B08)"
                        else: desc = f"Band {i}"
                    metadata["band_descriptions"].append(desc)

        except Exception as e:
            # Fallback using tifffile or PIL
            try:
                import tifffile
                with tifffile.TiffFile(file_path) as tf:
                    page = tf.pages[0]
                    metadata["height"], metadata["width"] = page.shape[:2]
                    metadata["bands"] = page.shape[2] if len(page.shape) > 2 else 1
                    metadata["has_nir"] = metadata["bands"] >= 4
            except Exception:
                from PIL import Image
                with Image.open(file_path) as img:
                    metadata["width"], metadata["height"] = img.size
                    metadata["bands"] = len(img.getbands())
    else:
        # Standard RGB image (PNG, JPG, JPEG)
        from PIL import Image
        with Image.open(file_path) as img:
            metadata["width"], metadata["height"] = img.size
            metadata["bands"] = len(img.getbands())

    # Build resolution transition label
    if metadata.get("pixel_resolution_m") is not None:
        src_res = metadata["pixel_resolution_m"]
        sr_res = round(src_res / 4.0, 2)
        metadata["resolution_label_template"] = f"Spatial resolution: {src_res} m/pixel → estimated {sr_res} m/pixel (4x SR)"
    else:
        metadata["resolution_label_template"] = "4x pixel upscaling applied; physical ground resolution requires validation."

    return metadata
