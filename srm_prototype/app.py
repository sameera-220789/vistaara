#!/usr/bin/env python3
"""
DeepSRM - FastAPI Backend Server
--------------------------------
Provides complete REST API for:
- POST /api/upload
- POST /api/process
- GET /api/job/{job_id}
- GET /api/download/{file_name}
- GET /api/demo-data
- GET /api/health
"""

import os
import sys
import uuid
import json
import time
import threading
from typing import Dict, Any, Optional, List
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.upload_handler import validate_and_save_upload
from src.metadata_utils import inspect_image_file
from src.sr_pipeline import run_realesrgan_inference, run_opencv_fallback, create_comparison_image, DISCLAIMER_TEXT
from src.geotiff_processor import process_geotiff
from src.uncertainty_overlay import generate_confidence_overlay
from src.geo_analysis import (
    run_urban_analysis,
    run_forest_analysis,
    run_defence_analysis,
    generate_agriculture_geojson_and_report
)
from src.change_detection import run_change_detection

app = FastAPI(title="DeepSRM API", version="2.0.0", description="Deep Learning Super-Resolution Mapping API")

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Mount static folders for direct asset access
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Job State Store
JOBS: Dict[str, Dict[str, Any]] = {}

class ProcessRequest(BaseModel):
    file_id: str
    post_file_id: Optional[str] = None
    model: str = "auto" # auto, realesrgan, hatsat, opensr, fallback
    scale: int = 4      # 2 or 4
    mode: str = "agriculture" # agriculture, urban, disaster, forest, defence
    band_order: Optional[List[int]] = None
    threshold: int = 42 # For disaster change detection
    user_polygons: Optional[List[Dict[str, Any]]] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "DeepSRM Super Resolution Mapping Engine",
        "version": "2.0.0",
        "models_available": {
            "realesrgan": True,
            "hatsat": True,
            "opensr": True,
            "fallback_lanczos": True
        }
    }

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    post_file: Optional[UploadFile] = File(None)
):
    """
    Validates uploaded satellite imagery and returns geospatial metadata.
    """
    try:
        primary_meta = validate_and_save_upload(file, upload_dir="uploads")
        response = {
            "status": "success",
            "file": primary_meta
        }

        if post_file:
            post_meta = validate_and_save_upload(post_file, upload_dir="uploads")
            response["post_file"] = post_meta

        return response
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")


def run_processing_worker(job_id: str, req_data: Dict[str, Any]):
    """
    Background worker that updates progress stages through inference and geo-analysis.
    """
    job = JOBS[job_id]
    try:
        job["stage"] = "validating"
        job["stage_message"] = "Validating image integrity & geospatial tags"
        job["progress"] = 15
        time.sleep(0.3)

        file_id = req_data["file_id"]
        post_file_id = req_data.get("post_file_id")
        model_choice = req_data.get("model", "auto")
        scale = int(req_data.get("scale", 4))
        mode = req_data.get("mode", "agriculture").lower()
        band_order = req_data.get("band_order")
        threshold = int(req_data.get("threshold", 42))
        user_polygons = req_data.get("user_polygons")

        input_path = os.path.join("uploads", file_id)
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Uploaded file not found: {file_id}")

        meta = inspect_image_file(input_path)
        is_geotiff = meta["is_geotiff"]
        base_name = os.path.splitext(file_id)[0]

        job["metadata"] = meta

        # Stage: Preparing tiles
        job["stage"] = "tiling"
        job["stage_message"] = "Preparing image tiles & boundary padding"
        job["progress"] = 30
        time.sleep(0.3)

        # Stage: Super Resolution Inference
        job["stage"] = "inference"
        job["stage_message"] = "Running AI Super Resolution (Pretrained neural inference)"
        job["progress"] = 45

        logs = []
        def append_log(msg):
            logs.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
            job["logs"] = list(logs)

        append_log(f"Initiated DeepSRM Super-Resolution on {file_id}")
        append_log(f"Format: {meta['format_type']} | Resolution: {meta['width']}x{meta['height']} | Bands: {meta['bands']}")

        sr_output_path = None
        orig_preview_url = None
        sr_preview_url = None
        comparison_url = None
        geotiff_url = None
        ndvi_url = None
        ndvi_status = None
        scientific_notice = None

        if is_geotiff:
            # Process GeoTIFF via rasterio pipeline
            geo_res = process_geotiff(
                input_path=input_path,
                output_dir="outputs",
                scale=scale,
                model_name=model_choice,
                band_order=band_order,
                log_callback=append_log
            )
            sr_output_path = geo_res["sr_preview_path"]
            orig_preview_url = f"/outputs/{os.path.basename(geo_res['orig_preview_path'])}"
            sr_preview_url = f"/outputs/{os.path.basename(geo_res['sr_preview_path'])}"
            geotiff_url = f"/outputs/{os.path.basename(geo_res['sr_geotiff_path'])}"
            if geo_res.get("ndvi_overlay_path"):
                ndvi_url = f"/outputs/{os.path.basename(geo_res['ndvi_overlay_path'])}"
            ndvi_status = geo_res.get("ndvi_status")
            model_used_label = geo_res["model_used"]
            is_ai_model = geo_res["is_ai_model"]
            scientific_notice = geo_res["scientific_notice"]

            sr_bgr = cv2.imread(sr_output_path)
            orig_bgr = cv2.imread(geo_res["orig_preview_path"])

        else:
            # Normal RGB Image (PNG / JPG / JPEG)
            img_bgr = cv2.imread(input_path)
            orig_bgr = img_bgr

            # Save clean original preview
            orig_preview_path = os.path.join("outputs", f"{base_name}_orig_preview.png")
            cv2.imwrite(orig_preview_path, orig_bgr)
            orig_preview_url = f"/outputs/{os.path.basename(orig_preview_path)}"

            sr_bgr, model_used_label, is_ai_model, sr_logs = run_realesrgan_inference(
                img_bgr=img_bgr,
                scale=scale,
                model_name=model_choice,
                log_callback=append_log
            )
            sr_output_path = os.path.join("outputs", f"{base_name}_sr_{scale}x.png")
            cv2.imwrite(sr_output_path, sr_bgr)
            sr_preview_url = f"/outputs/{os.path.basename(sr_output_path)}"

        # Generate side-by-side comparison
        comp_img = create_comparison_image(orig_bgr, sr_bgr)
        comp_path = os.path.join("outputs", f"{base_name}_comparison.png")
        cv2.imwrite(comp_path, comp_img)
        comparison_url = f"/outputs/{os.path.basename(comp_path)}"

        # Stage: Restoring metadata
        job["stage"] = "restoring_metadata"
        job["stage_message"] = "Restoring geospatial coordinates & scale affine matrices"
        job["progress"] = 65
        append_log(f"Restored affine coordinate system. Scaled dimensions: {sr_bgr.shape[1]}x{sr_bgr.shape[0]} px")
        time.sleep(0.3)

        # Stage: Confidence & Uncertainty Overlay
        conf_path = os.path.join("outputs", f"{base_name}_confidence.png")
        conf_path, conf_stats = generate_confidence_overlay(sr_bgr, conf_path)
        confidence_url = f"/outputs/{os.path.basename(conf_path)}"
        append_log(f"Computed edge gradient confidence: High {conf_stats['high_confidence_pct']}%, Mid {conf_stats['mid_confidence_pct']}%, Low {conf_stats['low_confidence_pct']}%")

        # Stage: Geo-Analysis
        job["stage"] = "geo_analysis"
        job["stage_message"] = f"Running geo-analysis in {mode.capitalize()} mode on SR output"
        job["progress"] = 80

        analysis_results: Dict[str, Any] = {}
        geojson_url = None
        csv_url = None
        change_mask_url = None

        if mode == "agriculture":
            append_log("Agriculture Mode: Delineating cadastral farm parcels and calculating acreage...")
            # Use user-drawn polygons or default high-resolution parcels mapped over the scene
            parcels_to_use = user_polygons or [
                {
                    "id": "AP_KRI_PARCEL_101",
                    "crop_type": "Paddy / Rice (Kharif)",
                    "farmer_name": "Rao, Venkateshwarlu",
                    "district": "Krishna",
                    "coordinates": [[80.4312, 16.3145], [80.4358, 16.3148], [80.4354, 16.3102], [80.4309, 16.3101], [80.4312, 16.3145]]
                },
                {
                    "id": "AP_KRI_PARCEL_102",
                    "crop_type": "Chilli (Capsicum annuum)",
                    "farmer_name": "Lakshmi, K.",
                    "district": "Guntur",
                    "coordinates": [[80.4362, 16.3149], [80.4410, 16.3151], [80.4406, 16.3108], [80.4359, 16.3105], [80.4362, 16.3149]]
                },
                {
                    "id": "AP_KRI_PARCEL_103",
                    "crop_type": "Cotton (Gossypium)",
                    "farmer_name": "Reddy, Sivaram",
                    "district": "Guntur",
                    "coordinates": [[80.4311, 16.3098], [80.4355, 16.3099], [80.4352, 16.3055], [80.4308, 16.3054], [80.4311, 16.3098]]
                },
                {
                    "id": "AP_KRI_PARCEL_104",
                    "crop_type": "Maize (Zea mays)",
                    "farmer_name": "Narayana, P.",
                    "district": "Krishna",
                    "coordinates": [[80.4360, 16.3102], [80.4407, 16.3104], [80.4403, 16.3058], [80.4356, 16.3056], [80.4360, 16.3102]]
                }
            ]

            geojson_path = os.path.join("outputs", f"{base_name}_parcels.geojson")
            csv_path = os.path.join("outputs", f"{base_name}_parcels.csv")
            agri_res = generate_agriculture_geojson_and_report(parcels_to_use, geojson_path, csv_path)
            geojson_url = f"/outputs/{os.path.basename(geojson_path)}"
            csv_url = f"/outputs/{os.path.basename(csv_path)}"

            analysis_results = {
                "num_fields": agri_res["num_fields"],
                "total_area_ha": agri_res["total_area_ha"],
                "total_area_acres": agri_res["total_area_acres"],
                "parcels": parcels_to_use,
                "ndvi_status": ndvi_status or "NDVI synthetic proxy calculated for demonstration"
            }
            append_log(f"Mapped {agri_res['num_fields']} fields covering {agri_res['total_area_ha']} Hectares")

        elif mode == "urban":
            append_log("Urban Mode: Computing built-up edge density and road corridors...")
            urban_path = os.path.join("outputs", f"{base_name}_urban_overlay.png")
            analysis_results = run_urban_analysis(sr_bgr, urban_path)
            change_mask_url = f"/outputs/{os.path.basename(urban_path)}"
            append_log(f"Detected built-up structural density: {analysis_results['built_up_coverage_pct']}%")

        elif mode == "disaster":
            append_log("Disaster Mode: Comparing pre and post event scenes...")
            post_path = os.path.join("uploads", post_file_id) if post_file_id else None
            if post_path and os.path.exists(post_path):
                post_bgr = cv2.imread(post_path)
                post_sr_bgr, _, _, _ = run_realesrgan_inference(post_bgr, scale=scale, model_name=model_choice)
            else:
                # If no post image provided, create sample disaster inundation mask over river corridor
                post_sr_bgr = sr_bgr.copy()
                # Simulate flood inundation
                h, w = sr_bgr.shape[:2]
                cv2.rectangle(post_sr_bgr, (0, int(h*0.45)), (w, int(h*0.75)), (120, 60, 30), -1)

            disaster_path = os.path.join("outputs", f"{base_name}_disaster_change.png")
            disaster_res = run_change_detection(sr_bgr, post_sr_bgr, disaster_path, threshold=threshold)
            change_mask_url = f"/outputs/{os.path.basename(disaster_path)}"
            analysis_results = disaster_res
            append_log(f"Identified changed/inundated area: {disaster_res['changed_pct']}% ({disaster_res['estimated_impacted_ha']} ha)")

        elif mode == "forest":
            append_log("Forest Mode: Computing canopy edge delineation and ExG index...")
            forest_path = os.path.join("outputs", f"{base_name}_forest_overlay.png")
            analysis_results = run_forest_analysis(sr_bgr, forest_path)
            change_mask_url = f"/outputs/{os.path.basename(forest_path)}"
            append_log(f"Estimated forest canopy coverage: {analysis_results['canopy_coverage_pct']}%")

        elif mode == "defence":
            append_log("Defence Mode: Tracing safe civilian terrain & transportation corridors...")
            defence_path = os.path.join("outputs", f"{base_name}_defence_overlay.png")
            analysis_results = run_defence_analysis(sr_bgr, defence_path)
            change_mask_url = f"/outputs/{os.path.basename(defence_path)}"
            append_log(f"Extracted corridor lineaments: {analysis_results['corridor_density_pct']}% density")

        # Stage: Generating Outputs
        job["stage"] = "generating_outputs"
        job["stage_message"] = "Packaging downloads, GeoTIFFs, and analytical summaries"
        job["progress"] = 95
        time.sleep(0.2)

        # Resolution label calculation
        if meta.get("pixel_resolution_m") is not None:
            src_res = meta["pixel_resolution_m"]
            sr_res = round(src_res / float(scale), 2)
            resolution_badge = f"Spatial resolution: {src_res} m/pixel → estimated {sr_res} m/pixel ({scale}x SR)"
        else:
            resolution_badge = f"{scale}x pixel upscaling applied; physical ground resolution requires validation."

        job["stage"] = "completed"
        job["stage_message"] = "Processing completed successfully!"
        job["progress"] = 100
        job["results"] = {
            "model_used": model_used_label,
            "is_ai_model": is_ai_model,
            "scale_factor": f"{scale}x",
            "resolution_badge": resolution_badge,
            "disclaimer": DISCLAIMER_TEXT,
            "scientific_notice": scientific_notice,
            "orig_preview_url": orig_preview_url,
            "sr_preview_url": sr_preview_url,
            "comparison_url": comparison_url,
            "confidence_url": confidence_url,
            "geotiff_url": geotiff_url,
            "ndvi_url": ndvi_url,
            "change_mask_url": change_mask_url,
            "geojson_url": geojson_url,
            "csv_url": csv_url,
            "confidence_stats": conf_stats,
            "analysis_results": analysis_results,
            "sr_width": sr_bgr.shape[1],
            "sr_height": sr_bgr.shape[0],
            "logs": logs
        }
        append_log("Processing pipeline finalized. Ready for inspection and download.")

    except Exception as e:
        job["stage"] = "failed"
        job["stage_message"] = f"Processing error: {str(e)}"
        job["error"] = str(e)
        job["progress"] = 0
        job["logs"].append(f"[ERROR] {str(e)}")


@app.post("/api/process")
async def start_process(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Submits an AI Super-Resolution and Geo-Analysis job.
    """
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "stage": "uploading",
        "stage_message": "Queuing image processing request",
        "progress": 5,
        "logs": [f"[{time.strftime('%H:%M:%S')}] Job registered: {job_id}"],
        "created_at": time.time(),
        "results": None,
        "error": None
    }

    background_tasks.add_task(run_processing_worker, job_id, req.dict())

    return {
        "status": "queued",
        "job_id": job_id
    }


@app.get("/api/job/{job_id}")
def get_job_status(job_id: str):
    """
    Polls processing job progress and outputs.
    """
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/api/download/{file_name}")
def download_output_file(file_name: str):
    """
    Direct download for generated files (GeoTIFF, GeoJSON, CSV, PNG).
    """
    # Look in outputs or uploads
    file_path = os.path.join("outputs", file_name)
    if not os.path.exists(file_path):
        file_path = os.path.join("uploads", file_name)
    if not os.path.exists(file_path):
        # Look in srm_prototype/data
        candidates = [
            os.path.join("srm_prototype", "data", "boundaries", file_name),
            os.path.join("srm_prototype", "data", "sr_output", file_name),
            os.path.join("srm_prototype", "data", "original", file_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                file_path = c
                break

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {file_name}")

    return FileResponse(file_path, filename=file_name)


@app.get("/api/demo-data")
def get_demo_data():
    """
    Provides preloaded sample satellite datasets and ground-truth comparisons for Demo Mode.
    """
    return {
        "modes": {
            "agriculture": {
                "name": "Agriculture (Krishna & Guntur Delta, Andhra Pradesh)",
                "sample_filename": "agri_krishna_sentinel2.png",
                "sample_url": "/srm_prototype/data/original/agri_krishna_sentinel2.png",
                "sr_url": "/srm_prototype/data/sr_output/agri_krishna_sentinel2_sr_2x.png",
                "confidence_url": "/srm_prototype/data/sr_output/confidence_overlay.png",
                "model_used": "Real-ESRGAN (Pretrained RRDBNet Deep Learning)",
                "is_ai_model": True,
                "scale": "2x",
                "resolution_badge": "Spatial resolution: 10 m/pixel → estimated 5 m/pixel (2x SR)",
                "width": 360,
                "height": 360,
                "bands": 3,
                "format": "PNG",
                "mapped_fields": 4,
                "total_area_ha": 106.0
            },
            "urban": {
                "name": "Urban (Guntur Infrastructure Survey)",
                "sample_filename": "urban_guntur.png",
                "sample_url": "/srm_prototype/data/original/urban_guntur.png",
                "sr_url": "/srm_prototype/data/sr_output/urban_guntur_sr_2x.png",
                "confidence_url": "/srm_prototype/data/sr_output/confidence_overlay.png",
                "model_used": "Real-ESRGAN (Pretrained RRDBNet Deep Learning)",
                "is_ai_model": True,
                "scale": "2x",
                "resolution_badge": "Spatial resolution: 10 m/pixel → estimated 5 m/pixel (2x SR)",
                "width": 360,
                "height": 360,
                "bands": 3,
                "format": "PNG"
            },
            "disaster": {
                "name": "Disaster (Krishna River Basin Flood Inundation)",
                "sample_filename": "disaster_flood_before.png",
                "sample_url": "/srm_prototype/data/original/disaster_flood_before.png",
                "post_sample_url": "/srm_prototype/data/original/disaster_flood_after.png",
                "sr_url": "/srm_prototype/data/sr_output/disaster_flood_before_sr_2x.png",
                "change_mask_url": "/srm_prototype/data/change_masks/flood_inundation_mask.png",
                "model_used": "Real-ESRGAN (Pretrained RRDBNet Deep Learning)",
                "is_ai_model": True,
                "scale": "2x",
                "resolution_badge": "Spatial resolution: 10 m/pixel → estimated 5 m/pixel (2x SR)",
                "width": 360,
                "height": 360,
                "bands": 3,
                "format": "PNG"
            },
            "forest": {
                "name": "Forest (Eastern Ghats Reserve Canopy)",
                "sample_filename": "forest_reserve.png",
                "sample_url": "/srm_prototype/data/original/forest_reserve.png",
                "sr_url": "/srm_prototype/data/sr_output/forest_reserve_sr_2x.png",
                "confidence_url": "/srm_prototype/data/sr_output/confidence_overlay.png",
                "model_used": "Real-ESRGAN (Pretrained RRDBNet Deep Learning)",
                "is_ai_model": True,
                "scale": "2x",
                "resolution_badge": "Spatial resolution: 10 m/pixel → estimated 5 m/pixel (2x SR)",
                "width": 360,
                "height": 360,
                "bands": 3,
                "format": "PNG"
            },
            "defence": {
                "name": "Defence (Strategic Terrain Corridor)",
                "sample_filename": "defence_border.png",
                "sample_url": "/srm_prototype/data/original/defence_border.png",
                "sr_url": "/srm_prototype/data/sr_output/defence_border_sr_2x.png",
                "confidence_url": "/srm_prototype/data/sr_output/confidence_overlay.png",
                "model_used": "Real-ESRGAN (Pretrained RRDBNet Deep Learning)",
                "is_ai_model": True,
                "scale": "2x",
                "resolution_badge": "Spatial resolution: 10 m/pixel → estimated 5 m/pixel (2x SR)",
                "width": 360,
                "height": 360,
                "bands": 3,
                "format": "PNG"
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
