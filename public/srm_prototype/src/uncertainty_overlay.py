#!/usr/bin/env python3
"""
DeepSRM - Heuristic Confidence & Uncertainty Overlay
---------------------------------------------------
DISCLAIMER: This script generates a heuristic confidence overlay based on
local spatial edge gradients and texture consistency. It is a visual prototype
demonstrator for hackathons and NOT a calibrated Bayesian / epistemic deep learning
uncertainty model.

Concept:
- High edge definition & sharp structural gradients (e.g. well-defined field bunds,
  canals, road corridors) -> GREEN (High Confidence >= 80%)
- Intermediate gradient textures -> YELLOW / AMBER (Medium Confidence 50-80%)
- Low contrast / ambiguous boundary blur -> RED (Lower Confidence < 50%)

Output:
- Saves overlay to `data/sr_output/confidence_overlay.png`

Usage:
    python src/uncertainty_overlay.py --input data/original/agri_krishna_sentinel2.png --output data/sr_output/confidence_overlay.png
"""

import os
import sys
import math
import argparse

def generate_confidence_map_opencv(input_path, output_path):
    import cv2
    import numpy as np

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read {input_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Sobel edge magnitude
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    # Normalize magnitude between 0 and 1
    norm_mag = cv2.normalize(magnitude, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)

    # 2. Smooth slightly to represent regional confidence zone
    blurred_conf = cv2.GaussianBlur(norm_mag, (9, 9), 2.5)

    # Invert/map: strong boundaries have high confidence, flat ambiguous zones have lower confidence
    # Mix base confidence (0.4) + edge boost (up to 0.6)
    conf_score = 0.4 + 0.6 * blurred_conf
    conf_score = np.clip(conf_score, 0.0, 1.0)

    # 3. Create Color Map: Red (low) -> Yellow (mid) -> Green (high)
    h, w = img.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8) # RGBA

    for y in range(h):
        for x in range(w):
            c = conf_score[y, x]
            if c > 0.65:
                # Green dominance
                r = int((1.0 - (c - 0.65) / 0.35) * 200)
                g = 220
                b = 40
            else:
                # Red / Yellow dominance
                r = 235
                g = int((c / 0.65) * 200)
                b = 30
            
            # Semi-transparent alpha (140 / 255)
            overlay[y, x] = [b, g, r, 140]

    cv2.imwrite(output_path, overlay)
    print(f"[Confidence Engine] Generated OpenCV gradient confidence map ({w}x{h})")

def generate_confidence_map_pure_python(input_path, output_path):
    import zlib
    import struct

    def read_png(path):
        with open(path, 'rb') as f:
            data = f.read()
        pos = 8
        w, h = 0, 0
        idat_chunks = []
        while pos < len(data):
            length, chunk_type = struct.unpack('>I4s', data[pos:pos+8])
            pos += 8
            chunk_data = data[pos:pos+length]
            pos += length + 4
            if chunk_type == b'IHDR':
                w, h = struct.unpack('>II', chunk_data[:8])
            elif chunk_type == b'IDAT':
                idat_chunks.append(chunk_data)
            elif chunk_type == b'IEND':
                break
        raw = zlib.decompress(b''.join(idat_chunks))
        stride = 1 + w * 3
        pixels = []
        for y in range(h):
            row_start = y * stride + 1
            row = []
            for x in range(w):
                idx = row_start + x * 3
                r = raw[idx]
                g = raw[idx+1]
                b = raw[idx+2]
                row.append((r, g, b))
            pixels.append(row)
        return w, h, pixels

    def write_rgba_png(path, w, h, rgba_pixels):
        raw_rows = []
        for y in range(h):
            row_bytes = bytearray([0])
            for x in range(w):
                r, g, b, a = rgba_pixels[y][x]
                row_bytes.extend([r, g, b, a])
            raw_rows.append(bytes(row_bytes))
        compressed = zlib.compress(b''.join(raw_rows), 6)

        def make_chunk(chunk_type, data):
            length = len(data)
            crc = zlib.crc32(chunk_type + data) & 0xffffffff
            return struct.pack('>I', length) + chunk_type + data + struct.pack('>I', crc)

        with open(path, 'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0))) # type 6 = RGBA
            f.write(make_chunk(b'IDAT', compressed))
            f.write(make_chunk(b'IEND', b''))

    w, h, pixels = read_png(input_path)
    rgba = []
    for y in range(h):
        row = []
        for x in range(w):
            # Calculate simple gradient difference with adjacent pixels
            r0, g0, b0 = pixels[y][x]
            r_right = pixels[y][min(x + 1, w - 1)][0]
            r_down = pixels[min(y + 1, h - 1)][x][0]
            diff = abs(int(r0) - int(r_right)) + abs(int(r0) - int(r_down))
            
            # Conf scale 0 to 1
            conf = min(1.0, 0.45 + (diff / 100.0) * 0.55)
            if conf > 0.65:
                # Green
                cr, cg, cb = 40, 220, 60
            elif conf > 0.5:
                # Amber / Yellow
                cr, cg, cb = 230, 200, 30
            else:
                # Red
                cr, cg, cb = 230, 50, 40
            row.append((cr, cg, cb, 140)) # 140 semi-transparent alpha
        rgba.append(row)

    write_rgba_png(output_path, w, h, rgba)
    print(f"[Confidence Engine] Generated pure-python gradient confidence map ({w}x{h})")

def main():
    parser = argparse.ArgumentParser(description="DeepSRM Heuristic Confidence / Uncertainty Overlay Generator")
    parser.add_argument("--input", default="data/original/agri_krishna_sentinel2.png", help="Path to input satellite image")
    parser.add_argument("--output", default="data/sr_output/confidence_overlay.png", help="Path to output overlay image")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"\n=======================================================")
    print(f" DeepSRM - Heuristic Spatial Confidence Generator      ")
    print(f"=======================================================")
    print(f" Input Image   : {args.input}")
    print(f" Output Overlay: {args.output}")
    print(f" [Notice] This is a prototype edge-gradient heuristic,")
    print(f"          not a calibrated Bayesian uncertainty model.")
    print(f"-------------------------------------------------------")

    if not os.path.exists(args.input):
        print(f"[DeepSRM Info] Input {args.input} not found yet. It will be generated during dummy data setup.")
        return

    try:
        generate_confidence_map_opencv(args.input, args.output)
    except Exception as e:
        print(f"[Fallback] OpenCV unavailable or failed ({e}). Using pure Python...")
        generate_confidence_map_pure_python(args.input, args.output)

    print(f"[Success] Confidence overlay written to: {args.output}")

if __name__ == "__main__":
    main()
