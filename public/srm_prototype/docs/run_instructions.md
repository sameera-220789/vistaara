# DeepSRM – Run Instructions & Setup Guide (Beginner Friendly)

This document provides step-by-step instructions for running the **DeepSRM** prototype on **Windows** (Command Prompt / PowerShell) and Linux/macOS.

---

## 1. Quick Start: Windows Commands (Step-by-Step)

Open **Command Prompt (`cmd`)** or **PowerShell** and navigate to the project directory:

```cmd
cd srm_prototype
```

### Step 1: Create and Activate a Python Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```
*(On Linux / macOS: `python3 -m venv venv && source venv/bin/activate`)*

### Step 2: Install Prototype Dependencies
```cmd
pip install -r requirements.txt
```
> **Note:** Even if PyTorch or OpenCV installation takes time or encounters network limits, the scripts feature built-in pure-Python fallbacks that execute without error.

### Step 3: Generate Dummy / Synthetic Satellite Imagery
Generates realistic medium-resolution satellite scenes for Agriculture, Urban, Disaster, Defence, and Forest:
```cmd
python src/generate_sample_data.py
```

### Step 4: Run Phase 1 – AI Super-Resolution Pipeline
Enhances all images from `data/original/` and saves 2x super-resolved outputs to `data/sr_output/`:
```cmd
python src/sr_pipeline.py --input data/original --output data/sr_output --scale 2
```

### Step 5: Run Phase 2 – Agricultural Geo-Analysis
Computes parcel geometries (Hectares, Acres, Perimeter) and generates GeoJSON and CSV outputs:
```cmd
python src/geo_analysis.py --output data/boundaries
```

### Step 6: Run Uncertainty / Confidence Overlay
Generates a spatial edge-gradient heuristic map (Green = High confidence, Red = Low confidence):
```cmd
python src/uncertainty_overlay.py --input data/original/agri_krishna_sentinel2.png --output data/sr_output/confidence_overlay.png
```

### Step 7: Run Change Detection (Disaster Flood Mode)
Compares pre-flood and post-flood imagery to calculate submerged acreage and generate a flood mask:
```cmd
python src/change_detection.py --before data/original/disaster_flood_before.png --after data/original/disaster_flood_after.png --output data/change_masks/flood_change_mask.png
```

### Step 8: Launch the Web Demo
You can open `web_demo/index.html` directly in any web browser (Chrome, Edge, Firefox):
```cmd
start web_demo\index.html
```
Or start a lightweight local web server:
```cmd
cd web_demo
python -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.

---

## 2. Optional: Pretrained Real-ESRGAN Deep Learning Setup

If you have an NVIDIA GPU and wish to execute neural inference using Real-ESRGAN rather than the interpolation fallback:

1. Install PyTorch with CUDA support:
   ```cmd
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install Real-ESRGAN and BasicSR:
   ```cmd
   pip install basicsr
   pip install realesrgan
   ```
3. Download the official pretrained model weights (`RealESRGAN_x2plus.pth` or `RealESRGAN_x4plus.pth`):
   ```cmd
   mkdir weights
   curl -L -o weights/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
   ```
4. Re-run the pipeline:
   ```cmd
   python src/sr_pipeline.py --input data/original --output data/sr_output --scale 2
   ```
   The engine will automatically detect PyTorch and run the neural network weights!

---

## 3. Alternative Workflow: Using QGIS for Field Boundary Delineation

You can digitize or inspect real farm plots in **QGIS** (Free & Open Source GIS):

### Step 1: Load Satellite Imagery in QGIS
1. Open QGIS Desktop.
2. Go to **Layer > Add Layer > Add Raster Layer...**
3. Select `data/sr_output/agri_krishna_sentinel2_sr_2x.png`.
4. (Optional) Set coordinate reference system to `EPSG:4326` or `EPSG:32644` (UTM Zone 44N for Andhra Pradesh).

### Step 2: Create a Field Boundary Vector Layer
1. Go to **Layer > Create Layer > New Shapefile Layer...** or **New GeoPackage Layer...**
2. Geometry type: **Polygon**.
3. Add fields:
   - `parcel_id` (Text)
   - `farmer_name` (Text)
   - `crop_type` (Text)
4. Click **OK** and save as `field_boundaries.shp`.

### Step 3: Digitize Parcels
1. Toggle Editing (Pencil icon).
2. Select **Add Polygon Feature** tool.
3. Trace the sharp field bunds visible in the super-resolved raster.
4. Right-click to complete polygon and enter the attribute values.
5. Save layer edits.

### Step 4: Export to GeoJSON for DeepSRM Web Demo
1. Right-click the polygon layer in the Layers panel > **Export > Save Features As...**
2. Format: **GeoJSON**.
3. File name: Select `srm_prototype/data/boundaries/field_boundaries.geojson`.
4. CRS: **EPSG:4326 - WGS 84**.
5. Click **OK**.
6. Refresh the DeepSRM web demo to visualize your digitized boundaries immediately!
