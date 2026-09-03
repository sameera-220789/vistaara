export interface ApplicationMode {
  id: "agriculture" | "urban" | "disaster" | "defence" | "forest";
  name: string;
  badge: string;
  icon: string;
  region: string;
  sensor: string;
  resolution: string;
  boundariesCount: number;
  mappedArea: string;
  govRelevance: string;
  description: string;
  originalImg: string;
  srImg: string;
  confidenceOverlay: string;
  changeMask?: string;
  parcels?: Array<{
    id: string;
    farmerName: string;
    cropType: string;
    district: string;
    areaHa: number;
    areaAcres: number;
    perimeterM: number;
    confidence: string;
    pmfbyStatus: string;
    coords: number[][];
  }>;
}

export const APPLICATION_MODES: Record<string, ApplicationMode> = {
  agriculture: {
    id: "agriculture",
    name: "Agriculture (Primary Working Demo)",
    badge: "Full Working Demo",
    icon: "Wheat",
    region: "Krishna & Guntur Districts, Andhra Pradesh",
    sensor: "Sentinel-2 MSI (10m Multi-Spectral)",
    resolution: "10m to 5m (2x Super-Resolution)",
    boundariesCount: 4,
    mappedArea: "106.0 Hectares (262.0 Acres)",
    govRelevance: "Clearer small field boundaries for FASAL, PMFBY and Krishi-DSS.",
    description: "Delineation of fragmented smallholder agricultural fields, irrigation canals, and narrow earthen bunds in the fertile Krishna Delta.",
    originalImg: "/srm_prototype/data/original/agri_krishna_sentinel2.png",
    srImg: "/srm_prototype/data/sr_output/agri_krishna_sentinel2_sr_2x.png",
    confidenceOverlay: "/srm_prototype/data/sr_output/confidence_overlay.png",
    parcels: [
      {
        id: "AP_KRI_PARCEL_101",
        farmerName: "Rao, Venkateshwarlu",
        cropType: "Paddy / Rice (Kharif)",
        district: "Krishna",
        areaHa: 24.19,
        areaAcres: 59.77,
        perimeterM: 1976,
        confidence: "94.2% (High)",
        pmfbyStatus: "Verified via DeepSRM",
        coords: [
          [80.4312, 16.3145],
          [80.4358, 16.3148],
          [80.4354, 16.3102],
          [80.4309, 16.3101],
          [80.4312, 16.3145]
        ]
      },
      {
        id: "AP_KRI_PARCEL_102",
        farmerName: "Lakshmi, K.",
        cropType: "Chilli (Capsicum annuum)",
        district: "Guntur",
        areaHa: 24.39,
        areaAcres: 60.26,
        perimeterM: 1986,
        confidence: "95.1% (High)",
        pmfbyStatus: "Verified via DeepSRM",
        coords: [
          [80.4362, 16.3149],
          [80.4410, 16.3151],
          [80.4406, 16.3108],
          [80.4359, 16.3105],
          [80.4362, 16.3149]
        ]
      },
      {
        id: "AP_KRI_PARCEL_103",
        farmerName: "Reddy, Srinivasa",
        cropType: "Cotton / Sugarcane",
        district: "Krishna",
        areaHa: 34.20,
        areaAcres: 84.51,
        perimeterM: 2380,
        confidence: "93.8% (High)",
        pmfbyStatus: "Verified via DeepSRM",
        coords: [
          [80.4310, 16.3096],
          [80.4375, 16.3098],
          [80.4372, 16.3054],
          [80.4307, 16.3051],
          [80.4310, 16.3096]
        ]
      },
      {
        id: "AP_KRI_PARCEL_104",
        farmerName: "Subbarao, M.",
        cropType: "Horticulture / Mango Grove",
        district: "Krishna",
        areaHa: 23.26,
        areaAcres: 57.47,
        perimeterM: 1940,
        confidence: "92.5% (High)",
        pmfbyStatus: "Verified via DeepSRM",
        coords: [
          [80.4380, 16.3097],
          [80.4428, 16.3099],
          [80.4425, 16.3058],
          [80.4377, 16.3056],
          [80.4380, 16.3097]
        ]
      }
    ]
  },
  urban: {
    id: "urban",
    name: "Urban (Infrastructure & SISDP-U)",
    badge: "Scalable Module",
    icon: "Building2",
    region: "Guntur City Municipal Corporation, Andhra Pradesh",
    sensor: "Sentinel-2 MSI (10m True Color)",
    resolution: "10m to 5m (2x Super-Resolution)",
    boundariesCount: 8,
    mappedArea: "62.4 Hectares (Built-up Fabric)",
    govRelevance: "Sharper settlement and infrastructure edges for SISDP-U and Urban Frame Survey.",
    description: "High-density road corridors, commercial complexes, and urban expansion fringe monitoring.",
    originalImg: "/srm_prototype/data/original/urban_guntur.png",
    srImg: "/srm_prototype/data/sr_output/urban_guntur_sr_2x.png",
    confidenceOverlay: "/srm_prototype/data/sr_output/confidence_overlay.png"
  },
  disaster: {
    id: "disaster",
    name: "Disaster (Flood Inundation & NDEM)",
    badge: "Scalable Module",
    icon: "Droplets",
    region: "Krishna River Lower Basin Inundation Zone",
    sensor: "Sentinel-2 MSI Pre/Post Flood Comparison",
    resolution: "10m to 5m (2x Super-Resolution)",
    boundariesCount: 3,
    mappedArea: "25.2 Hectares Submerged (62.3 Acres)",
    govRelevance: "Enhanced flood or landslide extent for NDEM and state disaster management portals.",
    description: "Rapid inundation assessment comparing baseline river levels against peak flood surge extent.",
    originalImg: "/srm_prototype/data/original/disaster_flood_before.png",
    srImg: "/srm_prototype/data/sr_output/disaster_flood_after_sr_2x.png",
    changeMask: "/srm_prototype/data/change_masks/flood_change_mask.png",
    confidenceOverlay: "/srm_prototype/data/sr_output/confidence_overlay.png"
  },
  defence: {
    id: "defence",
    name: "Defence (Strategic Terrain Corridor)",
    badge: "Scalable Module",
    icon: "ShieldAlert",
    region: "Demonstration Border Outpost & Track Corridor",
    sensor: "Medium Resolution Multispectral (Simulated Demo)",
    resolution: "10m to 5m (2x Super-Resolution)",
    boundariesCount: 2,
    mappedArea: "38.5 Hectares Terrain Corridor",
    govRelevance: "High-level terrain and infrastructure monitoring using non-sensitive demonstration data.",
    description: "Perimeter security lines, desert patrol corridors, and checkpoint footprint delineation.",
    originalImg: "/srm_prototype/data/original/defence_border.png",
    srImg: "/srm_prototype/data/sr_output/defence_border_sr_2x.png",
    confidenceOverlay: "/srm_prototype/data/sr_output/confidence_overlay.png"
  },
  forest: {
    id: "forest",
    name: "Forest (Canopy Edge & Conservation)",
    badge: "Scalable Module",
    icon: "Trees",
    region: "Eastern Ghats Forest Corridor, Andhra Pradesh",
    sensor: "Sentinel-2 MSI Near-Infrared / True Color",
    resolution: "10m to 5m (2x Super-Resolution)",
    boundariesCount: 5,
    mappedArea: "114.8 Hectares Forest Canopy",
    govRelevance: "Clearer forest edges and degradation indicators for forest departments.",
    description: "Delineation of reserve forest compartment boundaries, fire lines, and unauthorized clearings.",
    originalImg: "/srm_prototype/data/original/forest_reserve.png",
    srImg: "/srm_prototype/data/sr_output/forest_reserve_sr_2x.png",
    confidenceOverlay: "/srm_prototype/data/sr_output/confidence_overlay.png"
  }
};

export interface SRModelConfig {
  id: "sen2sr" | "realesrgan" | "hatsat" | "fallback";
  name: string;
  shortName: string;
  badge: string;
  architecture: string;
  nativeScale: string;
  targetResolution: string;
  spectralBands: string;
  radiometricConstraint: boolean;
  trainingDataset: string;
  description: string;
  accuracyMetric: string;
}

export const SR_MODELS: Record<string, SRModelConfig> = {
  sen2sr: {
    id: "sen2sr",
    name: "Sen2SR (ESA OpenSR Sentinel-2)",
    shortName: "Sen2SR 2.5m",
    badge: "Sentinel-2 Native",
    architecture: "RCAB / Hard-Constraint Residual Attention Network",
    nativeScale: "4x (10m → 2.5m)",
    targetResolution: "2.5m GSD",
    spectralBands: "Sentinel-2 MSI (B02 Blue, B03 Green, B04 Red, B08 NIR)",
    radiometricConstraint: true,
    trainingDataset: "SEN2NEON / ESA OpenSR High-Resolution Benchmark",
    description: "Purpose-built deep learning model for Sentinel-2 satellite imagery. Enforces strict radiometric consistency where downsampled 2.5m pixels strictly conserve original 10m surface reflectance.",
    accuracyMetric: "PSNR: 34.82 dB | SSIM: 0.912 | ΔE < 1.4%"
  },
  realesrgan: {
    id: "realesrgan",
    name: "Real-ESRGAN (RRDBNet)",
    shortName: "Real-ESRGAN",
    badge: "Edge Sharpener",
    architecture: "23-Block Residual-in-Residual Dense Network (RRDBNet)",
    nativeScale: "4x (Spatial Enhancement)",
    targetResolution: "Enhanced Crisp Edges",
    spectralBands: "3-Band RGB Visual Composite",
    radiometricConstraint: false,
    trainingDataset: "Synthetic Degradation Model & High-Resolution Imagery",
    description: "Deep convolutional network optimized for texture recovery, high-frequency edge restoration, and building footprint boundary delineation.",
    accuracyMetric: "PSNR: 31.45 dB | SSIM: 0.884"
  },
  hatsat: {
    id: "hatsat",
    name: "HATSAT (Hybrid Attention)",
    shortName: "HATSAT Prior",
    badge: "Remote Sensing",
    architecture: "Hybrid Attention Transformer with Window & Channel Self-Attention",
    nativeScale: "4x (Sensor Calibrated)",
    targetResolution: "High-Fidelity Feature Grid",
    spectralBands: "Multispectral RGB + Equalization Prior",
    radiometricConstraint: false,
    trainingDataset: "Remote Sensing Land-Cover Benchmarks",
    description: "Combines window-based self-attention with channel attention to capture both local geological contours and large-scale agricultural field structures.",
    accuracyMetric: "PSNR: 33.10 dB | SSIM: 0.898"
  },
  fallback: {
    id: "fallback",
    name: "Lanczos-4 Classical Interpolation",
    shortName: "Lanczos-4",
    badge: "Zero-AI Baseline",
    architecture: "8-Lobe Sinc Windowed Filter + Unsharp Masking",
    nativeScale: "2x / 4x",
    targetResolution: "Interpolated Grid",
    spectralBands: "All Available Bands",
    radiometricConstraint: true,
    trainingDataset: "None (Deterministic Mathematical Kernel)",
    description: "Standard mathematical resampling filter for guaranteed artifact-free inspection when neural network weights are offline.",
    accuracyMetric: "PSNR: 27.30 dB | SSIM: 0.792"
  }
};


