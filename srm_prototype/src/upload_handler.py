#!/usr/bin/env python3
"""
DeepSRM - Upload Handler & File Validation
------------------------------------------
Validates incoming file uploads (PNG, JPG, JPEG, TIF, TIFF).
Enforces security constraints, extracts geospatial metadata, and saves
files to local uploads/ workspace.
"""

import os
import uuid
import shutil
from typing import Dict, Any, Tuple
from fastapi import UploadFile, HTTPException

from src.metadata_utils import inspect_image_file

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024 # 100 MB limit

def validate_and_save_upload(
    upload_file: UploadFile,
    upload_dir: str = "uploads"
) -> Dict[str, Any]:
    """
    Saves an uploaded file to upload_dir and inspects its properties.
    """
    filename = upload_file.filename or "uploaded_image.png"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Accepted formats: PNG, JPG, JPEG, TIF, TIFF."
        )

    os.makedirs(upload_dir, exist_ok=True)
    file_id = f"{uuid.uuid4().hex[:10]}_{filename}"
    save_path = os.path.join(upload_dir, file_id)

    # Stream file to disk to prevent RAM saturation on large GeoTIFFs
    size = 0
    with open(save_path, "wb") as buffer:
        while True:
            chunk = upload_file.file.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                buffer.close()
                if os.path.exists(save_path):
                    os.remove(save_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)} MB."
                )
            buffer.write(chunk)

    # Inspect metadata
    try:
        metadata = inspect_image_file(save_path)
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(
            status_code=422,
            detail=f"Failed to read image raster data: {str(e)}"
        )

    metadata["stored_path"] = save_path
    metadata["file_id"] = file_id

    return metadata
