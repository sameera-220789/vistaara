# DeepSRM – Deep Learning Based Super Resolution Mapping from Medium Resolution Satellite Imagery

**Smart India Hackathon (SIH) Prototype Submission**  
**Theme:** Space Technology / Agriculture & Geospatial Intelligence  

---

## 1. Executive Summary

Medium-resolution earth observation satellites (such as **ISRO Resourcesat-2 LISS-IV / AWiFS** and **ESA Sentinel-2 at 10m Ground Sampling Distance**) provide open-access, frequent-revisit imagery over India. However, across India's agricultural belts—such as the **Krishna and Guntur delta in Andhra Pradesh**—the average landholding is small, fragmented, and bounded by narrow earthen bunds (0.5m to 2m wide).

At 10m/pixel, these bunds become blurry, creating severe uncertainty in crop acreage estimation, cadastral parcel mapping, and crop loss compensation. Very High Resolution (VHR) commercial satellites (WorldView, PlanetScope) cost millions of dollars for statewide coverage.

**DeepSRM (Deep Super-Resolution Mapping)** bridges this resolution gap using a cost-effective, two-phase AI geospatial approach:
1. **Phase 1 (AI Super-Resolution Engine):** Ingests medium-resolution satellite tiles and generates 2x to 4x enhanced imagery using deep convolutional / residual dense networks (pretrained Real-ESRGAN / ESRGAN architecture), with an automated CPU-compatible fallback.
2. **Phase 2 (Sub-Pixel Geo-Analysis & Vectorization):** Extracts sharp parcel geometries, computes geodesic metric areas (Hectares / Acres) and perimeters, and exports standardized GeoJSON and CSV datasets compatible with national portals (Krishi-DSS, PMFBY, FASAL).

---

## 2. Two-Phase Architecture Overview

```
 [Medium-Res Satellite Tile (10m)] (Sentinel-2 / LISS-IV)
                 │
                 ▼
 ┌────────────────────────────────────────────────────────┐
 │   PHASE 1: AI Super-Resolution Pipeline               │
 │   - Pretrained Real-ESRGAN (RRDBNet / Generative Prior)│
 │   - Fallback: Lanczos4 Interpolation + Unsharp Filter   │
 │   - Multi-scale support (2x, 4x upscaling)             │
 └────────────────────────────────────────────────────────┘
                 │
                 ▼
 [Super-Resolved Tile (5m / 2.5m effective sharpness)]
                 │
      ┌──────────┴────────────────────────┐
      ▼                                   ▼
 ┌─────────────────────────────┐   ┌─────────────────────────────┐
 │  PHASE 2: Geo-Analysis      │   │  Heuristic Confidence &     │
 │  - Precise Field Bunds      │   │  Change Detection           │
 │  - Geodesic Shoelace Area   │   │  - Spatial Edge Heuristic   │
 │  - GeoJSON & CSV Export     │   │  - Binary Disaster Mask     │
 └─────────────────────────────┘   └─────────────────────────────┘
      │                                   │
      ▼                                   ▼
 [Krishi-DSS / PMFBY Portals]        [NDEM / Disaster Portals]
```

### Phase 1: AI Super-Resolution (`src/sr_pipeline.py`)
- Reads raw PNG/JPG/TIFF satellite scenes from `data/original/`.
- Leverages a pretrained **Real-ESRGAN** deep learning model trained on complex degraded imagery.
- Features automatic fallbacks to high-order OpenCV interpolation (Lanczos-4 / Bicubic) with unsharp masking if PyTorch or GPU weights are unavailable.
- Saves high-clarity enhanced outputs to `data/sr_output/`.

### Phase 2: Geospatial Analysis & Boundary Delineation (`src/geo_analysis.py`)
- Digitizes and loads smallholder agricultural parcel polygons.
- Accurately computes **geodesic area** (Hectares, Acres, Square Meters) and **perimeter** (Meters) using the spherical Shoelace algorithm and Haversine distance.
- Exports industry-standard GeoJSON to `data/boundaries/field_boundaries.geojson` and summary metrics to `data/boundaries/analysis_summary.csv`.
- Directly ingestible into QGIS, ArcGIS, Google Earth Engine, or Krishi-DSS.

---

## 3. Multi-Sector Applications

While **Agriculture (Krishna/Guntur District, Andhra Pradesh)** serves as the primary end-to-end working demonstration, DeepSRM is modularly architected for 5 vital national domains:

| Sector | Primary Objective | Key Government Initiative Supported |
|---|---|---|
| 🌾 **Agriculture** (Full Demo) | Clearer small field boundaries, bund delineation, crop stage monitoring | **PMFBY** (Pradhan Mantri Fasal Bima Yojana), **FASAL**, **Krishi-DSS** |
| 🏙️ **Urban** | Sharper settlement edges, unauthorized construction detection, road corridor extraction | **SISDP-U** (Space-based Information Support for Decentralised Planning - Urban), Urban Frame Survey |
| 🌊 **Disaster** | Rapid flood inundation extent mapping, breach detection, landslide perimeter mapping | **NDEM** (National Database for Emergency Management), SDMA disaster portals |
| 🛡️ **Defence** | High-level terrain and infrastructure surveillance using non-sensitive demo data | Border track monitoring, logistics corridor assessment |
| 🌲 **Forest** | Delineation of forest compartment lines, firebreak corridors, encroachment detection | State Forest Departments, ISRO Bhuvan Forestry Services |

---

## 4. Key Prototype Features

- **Draggable Before/After Split Slider:** Interactive comparison of 10m medium-resolution original vs. 2x super-resolved output in real time.
- **Interactive GeoJSON Layer:** Vector polygons rendered over imagery with hover cards displaying parcel IDs, farmer names, crop categories, hectare acreage, and PMFBY audit status.
- **Spatial Confidence Overlay:** Prototype heuristic map highlighting high-contrast bund structures (Green) vs ambiguous zones (Amber/Red).
- **Automated Change Detection Engine:** Pre- and post-flood radiometric difference masking for disaster management.
- **Zero-Crash Architecture:** CPU-friendly fallback mechanisms ensure that the demo runs in any environment without specialized hardware.
