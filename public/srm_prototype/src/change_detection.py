#!/usr/bin/env python3
"""
DeepSRM - Change Detection & Disaster Impact Masking
---------------------------------------------------
Problem Statement: During severe climate events (e.g. Krishna river basin floods,
cyclonic storm surges in coastal AP), rapid damage assessment requires comparing
pre-event and post-event satellite scenes to isolate submerged croplands and
inundated rural settlements.

This script:
1. Compares a 'before' satellite image and an 'after' satellite image using
   absolute radiometric pixel difference.
2. Applies adaptive thresholding to produce a binary Change Mask.
3. Generates an annotated colored overlay highlighting new water inundation (Cyan / Red).
4. Computes quantitative change metrics:
   - Total scene pixels
   - Changed/Inundated pixels
   - Approximate impacted area in Hectares / Square Kilometers.
5. Saves results to `data/change_masks/` to support the Disaster Mode concept.

Usage:
    python src/change_detection.py --before data/original/disaster_flood_before.png --after data/original/disaster_flood_after.png --output data/change_masks/flood_change_mask.png
"""

import os
import sys
import argparse

def run_change_detection_opencv(before_path, after_path, output_path, threshold=40):
    import cv2
    import numpy as np

    img_before = cv2.imread(before_path)
    img_after = cv2.imread(after_path)

    if img_before is None or img_after is None:
        raise ValueError("Could not read before or after image files")

    # Resize to match dimensions if needed
    if img_before.shape != img_after.shape:
        img_after = cv2.resize(img_after, (img_before.shape[1], img_before.shape[0]))

    gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
    gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)

    # 1. Absolute difference
    diff = cv2.absdiff(gray_before, gray_after)

    # 2. Binary threshold
    _, thresh_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 3. Morphological cleanup (remove salt-and-pepper noise)
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel)

    # Calculate statistics
    total_pixels = clean_mask.size
    changed_pixels = np.count_nonzero(clean_mask)
    change_pct = (changed_pixels / total_pixels) * 100.0

    # Assume each pixel is approx 5m x 5m = 25 sqm (Super-resolved Sentinel-2)
    pixel_area_sqm = 25.0
    impacted_sqm = changed_pixels * pixel_area_sqm
    impacted_hectares = impacted_sqm / 10000.0

    # 4. Create visual RGBA flood inundation mask (Vibrant Cyan / Red)
    h, w = clean_mask.shape
    colored_overlay = np.zeros((h, w, 4), dtype=np.uint8)
    # Mask active areas: Bright Red/Cyan with 180 alpha
    colored_overlay[clean_mask > 0] = [235, 70, 30, 200]  # BGRA (Red-Orange alert)

    cv2.imwrite(output_path, colored_overlay)

    # Also save raw binary mask
    binary_path = output_path.replace(".png", "_binary.png")
    cv2.imwrite(binary_path, clean_mask)

    return total_pixels, changed_pixels, change_pct, impacted_hectares

def run_change_detection_pure_python(before_path, after_path, output_path, threshold=40):
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
            f.write(make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
            f.write(make_chunk(b'IDAT', compressed))
            f.write(make_chunk(b'IEND', b''))

    w1, h1, px1 = read_png(before_path)
    w2, h2, px2 = read_png(after_path)
    w = min(w1, w2)
    h = min(h1, h2)

    total_pixels = w * h
    changed_pixels = 0
    rgba = []

    for y in range(h):
        row = []
        for x in range(w):
            g1 = int(0.299 * px1[y][x][0] + 0.587 * px1[y][x][1] + 0.114 * px1[y][x][2])
            g2 = int(0.299 * px2[y][x][0] + 0.587 * px2[y][x][1] + 0.114 * px2[y][x][2])
            diff = abs(g1 - g2)
            if diff > threshold:
                changed_pixels += 1
                row.append((240, 50, 40, 200)) # Inundation alert color
            else:
                row.append((0, 0, 0, 0)) # Transparent
        rgba.append(row)

    write_rgba_png(output_path, w, h, rgba)
    change_pct = (changed_pixels / total_pixels) * 100.0
    impacted_hectares = (changed_pixels * 25.0) / 10000.0
    return total_pixels, changed_pixels, change_pct, impacted_hectares

def main():
    parser = argparse.ArgumentParser(description="DeepSRM Disaster Change Detection Engine")
    parser.add_argument("--before", default="data/original/disaster_flood_before.png", help="Pre-event satellite image")
    parser.add_argument("--after", default="data/original/disaster_flood_after.png", help="Post-event satellite image")
    parser.add_argument("--output", default="data/change_masks/flood_change_mask.png", help="Output change mask path")
    parser.add_argument("--threshold", type=int, default=40, help="Pixel difference threshold (0-255)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    print(f"\n=======================================================")
    print(f" DeepSRM - Change Detection (Disaster Response Engine) ")
    print(f"=======================================================")
    print(f" Pre-Event Image  : {args.before}")
    print(f" Post-Event Image : {args.after}")
    print(f" Output Mask      : {args.output}")
    print(f" Difference Delta : {args.threshold}")
    print(f"-------------------------------------------------------")

    if not os.path.exists(args.before) or not os.path.exists(args.after):
        print(f"[DeepSRM Info] Input images not yet found. They will be generated during dummy data setup.")
        return

    try:
        total, changed, pct, ha = run_change_detection_opencv(args.before, args.after, args.output, args.threshold)
    except Exception as e:
        print(f"[Fallback] OpenCV unavailable ({e}). Using pure Python...")
        total, changed, pct, ha = run_change_detection_pure_python(args.before, args.after, args.output, args.threshold)

    print(f"[Results]")
    print(f" Total Spatial Pixels: {total:,}")
    print(f" Inundated / Changed : {changed:,} pixels ({pct:.2f}%)")
    print(f" Estimated Impact Area: {ha:.2f} Hectares (~{ha * 2.471:.2f} Acres)")
    print(f"[Output] Change mask saved to {args.output}")

if __name__ == "__main__":
    main()
