#!/usr/bin/env python3
"""
DeepSRM - Super Resolution Pipeline (Phase 1)
--------------------------------------------
Problem Statement: Medium-resolution satellite imagery (e.g. Sentinel-2 at 10m/px)
often blurs narrow field bunds, small farm boundaries, and road edges in rural India.

This script enhances satellite images by 2x or 4x:
1. Attempts to use a Pretrained Deep Learning Super-Resolution model (Real-ESRGAN).
2. If PyTorch / Real-ESRGAN is not installed or GPU/weights are missing,
   it seamlessly falls back to OpenCV Bicubic / Lanczos upscaling.
3. If OpenCV is also absent, a lightweight pure-Python image resampler is used.
This guarantees the SIH prototype always runs and never crashes!

Usage:
    python src/sr_pipeline.py --input data/original --output data/sr_output --scale 2
"""

import os
import sys
import argparse
import glob
import time

# Helper function to check if Real-ESRGAN is available
def load_realesrgan_model(scale=2):
    """
    Attempts to import and initialize Real-ESRGAN model.
    Returns (model, model_type) or (None, None).
    """
    try:
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        print("[AI Engine] Checking for Real-ESRGAN dependencies...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AI Engine] Using device: {device}")

        # Standard RealESRGAN_x2plus or RealESRGAN_x4plus architecture
        if scale == 2:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            model_name = 'RealESRGAN_x2plus'
        else:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            model_name = 'RealESRGAN_x4plus'

        upsampler = RealESRGANer(
            scale=scale,
            model_path=None, # will download weights or use default if configured
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device=device
        )
        print(f"[AI Engine] Successfully loaded Real-ESRGAN ({model_name})")
        return upsampler, "Real-ESRGAN"
    except Exception as e:
        print(f"[AI Engine Info] Real-ESRGAN deep model not loaded ({e}).")
        print("[AI Engine Info] Switching to high-quality OpenCV Interpolation fallback.")
        return None, None

def upscale_with_opencv(img_path, output_path, scale=2):
    """
    Upscales an image using OpenCV with Lanczos/Cubic interpolation.
    Simulates enhanced contrast and edge sharpening typical of satellite preprocessing.
    """
    import cv2
    import numpy as np

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)

    # 1. High-fidelity interpolation
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 2. Subtle unsharp masking to enhance field boundaries & ridge lines
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.35, gaussian, -0.35, 0)

    cv2.imwrite(output_path, sharpened)
    return new_w, new_h

def pure_python_png_resample(img_path, output_path, scale=2):
    """
    Zero-dependency pure Python PNG/PPM scaler for environments without OpenCV.
    Reads and writes uncompressed/zlib PNGs using Python standard library.
    """
    import zlib
    import struct

    def read_png(path):
        with open(path, 'rb') as f:
            data = f.read()
        if data[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError("Not a valid PNG file")
        pos = 8
        w, h = 0, 0
        idat_chunks = []
        while pos < len(data):
            length, chunk_type = struct.unpack('>I4s', data[pos:pos+8])
            pos += 8
            chunk_data = data[pos:pos+length]
            pos += length + 4 # skip crc
            if chunk_type == b'IHDR':
                w, h, bit_depth, color_type = struct.unpack('>IIBB', chunk_data[:10])
            elif chunk_type == b'IDAT':
                idat_chunks.append(chunk_data)
            elif chunk_type == b'IEND':
                break
        
        raw_decompressed = zlib.decompress(b''.join(idat_chunks))
        # Parse uncompressed RGB pixels
        stride = 1 + w * 3
        pixels = []
        for y in range(h):
            row_start = y * stride + 1
            row = []
            for x in range(w):
                idx = row_start + x * 3
                r = raw_decompressed[idx]
                g = raw_decompressed[idx+1]
                b = raw_decompressed[idx+2]
                row.append((r, g, b))
            pixels.append(row)
        return w, h, pixels

    def write_png(path, w, h, pixels):
        raw_rows = []
        for y in range(h):
            row_bytes = bytearray([0]) # Filter type 0 (None)
            for x in range(w):
                r, g, b = pixels[y][x]
                row_bytes.extend([min(255, max(0, int(r))), min(255, max(0, int(g))), min(255, max(0, int(b)))])
            raw_rows.append(bytes(row_bytes))
        
        compressed = zlib.compress(b''.join(raw_rows), 6)
        
        def make_chunk(chunk_type, data):
            length = len(data)
            crc = zlib.crc32(chunk_type + data) & 0xffffffff
            return struct.pack('>I', length) + chunk_type + data + struct.pack('>I', crc)

        png_header = b'\x89PNG\r\n\x1a\n'
        ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
        
        with open(path, 'wb') as f:
            f.write(png_header)
            f.write(make_chunk(b'IHDR', ihdr_data))
            f.write(make_chunk(b'IDAT', compressed))
            f.write(make_chunk(b'IEND', b''))

    w, h, pixels = read_png(img_path)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Bilinear interpolation
    new_pixels = []
    for ny in range(new_h):
        row = []
        gy = ny / scale
        y0 = int(gy)
        y1 = min(y0 + 1, h - 1)
        wy = gy - y0
        for nx in range(new_w):
            gx = nx / scale
            x0 = int(gx)
            x1 = min(x0 + 1, w - 1)
            wx = gx - x0
            
            # 4 neighbors
            c00 = pixels[y0][x0]
            c10 = pixels[y0][x1]
            c01 = pixels[y1][x0]
            c11 = pixels[y1][x1]
            
            rgb = []
            for c in range(3):
                top = c00[c] * (1 - wx) + c10[c] * wx
                bottom = c01[c] * (1 - wx) + c11[c] * wx
                val = top * (1 - wy) + bottom * wy
                rgb.append(val)
            row.append((rgb[0], rgb[1], rgb[2]))
        new_pixels.append(row)

    write_png(output_path, new_w, new_h, new_pixels)
    return new_w, new_h

def process_pipeline(input_dir, output_dir, scale=2):
    """
    Runs the Super-Resolution enhancement pipeline for all satellite images in input_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images
    extensions = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff')
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_paths:
        print(f"[DeepSRM Warning] No images found in {input_dir}. Please add sample imagery.")
        return

    print(f"\n=======================================================")
    print(f" DeepSRM - Phase 1: Satellite Super-Resolution Engine ")
    print(f"=======================================================")
    print(f" Input Directory  : {input_dir}")
    print(f" Output Directory : {output_dir}")
    print(f" Target Scale     : {scale}x")
    print(f" Images Found     : {len(image_paths)}")
    print(f"-------------------------------------------------------")

    # Step 1: Check for Real-ESRGAN
    realesrgan_model, engine_type = load_realesrgan_model(scale)

    success_count = 0
    for idx, img_path in enumerate(image_paths, 1):
        filename = os.path.basename(img_path)
        base, ext = os.path.splitext(filename)
        output_filename = f"{base}_sr_{scale}x.png"
        output_path = os.path.join(output_dir, output_filename)

        print(f"\n[{idx}/{len(image_paths)}] Processing '{filename}'...")
        start_time = time.time()

        if realesrgan_model is not None:
            try:
                import cv2
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                output, _ = realesrgan_model.enhance(img, outscale=scale)
                cv2.imwrite(output_path, output)
                elapsed = time.time() - start_time
                print(f" -> Enhanced via Real-ESRGAN in {elapsed:.2f}s")
                print(f" -> Saved to: {output_path}")
                success_count += 1
                continue
            except Exception as e:
                print(f" -> Real-ESRGAN inference failed ({e}). Falling back to OpenCV.")

        # Step 2: Try OpenCV
        try:
            new_w, new_h = upscale_with_opencv(img_path, output_path, scale)
            elapsed = time.time() - start_time
            print(f" -> Upscaled via OpenCV Lanczos+Sharpening ({scale}x) to {new_w}x{new_h} in {elapsed:.2f}s")
            print(f" -> Saved to: {output_path}")
            success_count += 1
            continue
        except ImportError:
            print(" -> OpenCV not found in environment. Using pure-Python PNG fallback.")
        except Exception as e:
            print(f" -> OpenCV error: {e}. Trying pure-Python fallback.")

        # Step 3: Pure Python Fallback (Guaranteed to execute)
        try:
            new_w, new_h = pure_python_png_resample(img_path, output_path, scale)
            elapsed = time.time() - start_time
            print(f" -> Upscaled via Pure Python Resampler ({scale}x) to {new_w}x{new_h} in {elapsed:.2f}s")
            print(f" -> Saved to: {output_path}")
            success_count += 1
        except Exception as e:
            print(f" -> [Error] Failed processing {filename}: {e}")

    print(f"\n[DeepSRM Complete] Successfully processed {success_count}/{len(image_paths)} images.")

def main():
    parser = argparse.ArgumentParser(description="DeepSRM Phase 1 - Satellite Imagery Super-Resolution")
    parser.add_argument("--input", default="data/original", help="Path to original input images folder")
    parser.add_argument("--output", default="data/sr_output", help="Path to save super-resolved outputs")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4], help="Upscaling factor (2 or 4)")
    args = parser.parse_args()

    # Resolve relative paths from current working directory
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    process_pipeline(input_path, output_path, args.scale)

if __name__ == "__main__":
    main()
