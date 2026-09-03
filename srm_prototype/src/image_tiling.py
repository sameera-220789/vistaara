#!/usr/bin/env python3
"""
DeepSRM - Image Tiling & Seamless Blending Module
-------------------------------------------------
Splits large satellite imagery (>1024x1024 or user-configured size) into overlapping
tiles to prevent Out-Of-Memory (OOM) errors during Deep Learning Super-Resolution inference.
Tiles are super-resolved individually and blended seamlessly using cosine/linear window weighting.
"""

import numpy as np
import cv2

def create_blend_weight_mask(tile_h, tile_w, overlap):
    """
    Creates a 2D weight mask with smooth cosine rolloff in the overlap regions.
    """
    wx = np.ones(tile_w, dtype=np.float32)
    wy = np.ones(tile_h, dtype=np.float32)

    if overlap > 0:
        # Cosine ramp from 0 to 1 over overlap distance
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap, dtype=np.float32))
        wx[:overlap] = ramp
        wx[-overlap:] = ramp[::-1]
        wy[:overlap] = ramp
        wy[-overlap:] = ramp[::-1]

    mask = np.outer(wy, wx)
    return mask

def process_image_with_tiling(image_bgr, sr_func, scale=4, tile_size=512, overlap=48):
    """
    Processes an image in overlapping tiles if its dimensions exceed tile_size.
    sr_func: A callable that takes a BGR numpy array (tile_h, tile_w, 3) and returns (tile_h*scale, tile_w*scale, 3)
    """
    h, w = image_bgr.shape[:2]

    # If image is small enough, process directly
    if h <= tile_size and w <= tile_size:
        return sr_func(image_bgr)

    stride = tile_size - overlap
    out_h = h * scale
    out_w = w * scale
    channels = image_bgr.shape[2] if len(image_bgr.shape) == 3 else 1

    accum_output = np.zeros((out_h, out_w, channels), dtype=np.float32)
    accum_weights = np.zeros((out_h, out_w, 1), dtype=np.float32)

    y_steps = list(range(0, h, stride))
    x_steps = list(range(0, w, stride))

    # Adjust last steps to cover the edges without exceeding dimensions
    if y_steps[-1] + tile_size < h:
        y_steps.append(h - tile_size)
    else:
        y_steps[-1] = max(0, h - tile_size)

    if x_steps[-1] + tile_size < w:
        x_steps.append(w - tile_size)
    else:
        x_steps[-1] = max(0, w - tile_size)

    y_steps = sorted(list(set(y_steps)))
    x_steps = sorted(list(set(x_steps)))

    for y in y_steps:
        for x in x_steps:
            cur_tile_h = min(tile_size, h - y)
            cur_tile_w = min(tile_size, w - x)
            tile = image_bgr[y:y + cur_tile_h, x:x + cur_tile_w]

            # Run super-resolution on tile
            sr_tile = sr_func(tile).astype(np.float32)
            if sr_tile.ndim == 2:
                sr_tile = sr_tile[:, :, np.newaxis]

            # Generate blend weight mask for this tile size
            sr_tile_h, sr_tile_w = sr_tile.shape[:2]
            scaled_overlap = overlap * scale
            weight_mask = create_blend_weight_mask(sr_tile_h, sr_tile_w, scaled_overlap)
            weight_mask = weight_mask[:, :, np.newaxis]

            out_y = y * scale
            out_x = x * scale

            accum_output[out_y:out_y + sr_tile_h, out_x:out_x + sr_tile_w] += sr_tile * weight_mask
            accum_weights[out_y:out_y + sr_tile_h, out_x:out_x + sr_tile_w] += weight_mask

    # Normalize accumulated values by weights
    accum_weights = np.maximum(accum_weights, 1e-6)
    result = accum_output / accum_weights
    result = np.clip(result, 0, 255).astype(np.uint8)

    if channels == 1 and result.shape[2] == 1:
        result = result.squeeze(axis=2)

    return result
