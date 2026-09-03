/**
 * DeepSRM - AI Super-Resolution Mapping Platform
 * Vanilla JavaScript Web Demo Controller
 */

// Embedded fallback configs (guarantees local double-click file:// execution without CORS failure)
const DEFAULT_CONFIGS = {
  agriculture: {
    mode: "agriculture",
    title: "Agriculture (Primary Working Demo)",
    region: "Krishna & Guntur Districts, Andhra Pradesh, India",
    satellite_sensor: "Sentinel-2 MSI (10m Multi-Spectral)",
    resolution_enhancement: "10m to 5m (2x Super-Resolution)",
    boundaries_count: 4,
    mapped_area: "106.0 Hectares (262.0 Acres)",
    gov_relevance: "Clearer small field boundaries for FASAL, PMFBY and Krishi-DSS.",
    original_image: "../data/original/agri_krishna_sentinel2.png",
    sr_image: "../data/sr_output/agri_krishna_sentinel2_sr_2x.png",
    confidence_overlay: "../data/sr_output/confidence_overlay.png",
    change_mask: null,
    geojson_data: {
      type: "FeatureCollection",
      features: [
        {
          type: "Feature",
          properties: {
            parcel_id: "AP_KRI_PARCEL_101",
            farmer_name: "Rao, Venkateshwarlu",
            crop_type: "Paddy / Rice (Kharif)",
            area_ha: 24.18,
            area_acres: 59.76,
            perimeter_m: 1975
          },
          geometry: {
            type: "Polygon",
            coordinates: [[[80.4312, 16.3145], [80.4358, 16.3148], [80.4354, 16.3102], [80.4309, 16.3101], [80.4312, 16.3145]]]
          }
        },
        {
          type: "Feature",
          properties: {
            parcel_id: "AP_KRI_PARCEL_102",
            farmer_name: "Lakshmi, K.",
            crop_type: "Chilli (Capsicum annuum)",
            area_ha: 24.38,
            area_acres: 60.26,
            perimeter_m: 1985
          },
          geometry: {
            type: "Polygon",
            coordinates: [[[80.4362, 16.3149], [80.4410, 16.3151], [80.4406, 16.3108], [80.4359, 16.3105], [80.4362, 16.3149]]]
          }
        },
        {
          type: "Feature",
          properties: {
            parcel_id: "AP_KRI_PARCEL_103",
            farmer_name: "Reddy, Srinivasa",
            crop_type: "Cotton / Sugarcane",
            area_ha: 34.19,
            area_acres: 84.50,
            perimeter_m: 2380
          },
          geometry: {
            type: "Polygon",
            coordinates: [[[80.4310, 16.3096], [80.4375, 16.3098], [80.4372, 16.3054], [80.4307, 16.3051], [80.4310, 16.3096]]]
          }
        },
        {
          type: "Feature",
          properties: {
            parcel_id: "AP_KRI_PARCEL_104",
            farmer_name: "Subbarao, M.",
            crop_type: "Horticulture / Mango Grove",
            area_ha: 23.25,
            area_acres: 57.46,
            perimeter_m: 1939
          },
          geometry: {
            type: "Polygon",
            coordinates: [[[80.4380, 16.3097], [80.4428, 16.3099], [80.4425, 16.3058], [80.4377, 16.3056], [80.4380, 16.3097]]]
          }
        }
      ]
    }
  },
  urban: {
    mode: "urban",
    title: "Urban Infrastructure & Settlement Mapping",
    region: "Guntur Municipal Corporation, Andhra Pradesh",
    satellite_sensor: "Sentinel-2 MSI (10m True Color)",
    resolution_enhancement: "10m to 5m (2x Super-Resolution)",
    boundaries_count: 8,
    mapped_area: "62.4 Hectares (Built-up Fabric)",
    gov_relevance: "Sharper settlement and infrastructure edges for SISDP-U and Urban Frame Survey.",
    original_image: "../data/original/urban_guntur.png",
    sr_image: "../data/sr_output/urban_guntur_sr_2x.png",
    confidence_overlay: "../data/sr_output/confidence_overlay.png",
    change_mask: null,
    geojson_data: null
  },
  disaster: {
    mode: "disaster",
    title: "Disaster Response & Flood Inundation Assessment",
    region: "Krishna River Lower Basin Inundation Zone",
    satellite_sensor: "Sentinel-2 MSI Pre/Post Flood Comparison",
    resolution_enhancement: "10m to 5m (2x Super-Resolution)",
    boundaries_count: 3,
    mapped_area: "25.2 Hectares Submerged (62.3 Acres)",
    gov_relevance: "Enhanced flood or landslide extent for NDEM and state disaster management portals.",
    original_image: "../data/original/disaster_flood_before.png",
    sr_image: "../data/sr_output/disaster_flood_after_sr_2x.png",
    change_mask: "../data/change_masks/flood_change_mask.png",
    confidence_overlay: "../data/sr_output/confidence_overlay.png",
    geojson_data: null
  },
  defence: {
    mode: "defence",
    title: "Strategic Terrain & Perimeter Monitoring",
    region: "Demonstration Border Outpost & Track Corridor",
    satellite_sensor: "Medium Resolution Multispectral (Simulated)",
    resolution_enhancement: "10m to 5m (2x Super-Resolution)",
    boundaries_count: 2,
    mapped_area: "38.5 Hectares Terrain Corridor",
    gov_relevance: "High-level terrain and infrastructure monitoring using non-sensitive demonstration data.",
    original_image: "../data/original/defence_border.png",
    sr_image: "../data/sr_output/defence_border_sr_2x.png",
    confidence_overlay: "../data/sr_output/confidence_overlay.png",
    change_mask: null,
    geojson_data: null
  },
  forest: {
    mode: "forest",
    title: "Forest Reserve & Canopy Edge Conservation",
    region: "Eastern Ghats Forest Corridor, Andhra Pradesh",
    satellite_sensor: "Sentinel-2 MSI Near-Infrared / True Color",
    resolution_enhancement: "10m to 5m (2x Super-Resolution)",
    boundaries_count: 5,
    mapped_area: "114.8 Hectares Forest Canopy",
    gov_relevance: "Clearer forest edges and degradation indicators for forest departments.",
    original_image: "../data/original/forest_reserve.png",
    sr_image: "../data/sr_output/forest_reserve_sr_2x.png",
    confidence_overlay: "../data/sr_output/confidence_overlay.png",
    change_mask: null,
    geojson_data: null
  }
};

let currentMode = "agriculture";
let currentConfig = DEFAULT_CONFIGS.agriculture;
let isDragging = false;
let sliderPercent = 50;

// DOM Elements
const modeSelect = document.getElementById("modeSelect");
const imgOriginal = document.getElementById("img-original");
const imgSR = document.getElementById("img-sr");
const overlayConfidence = document.getElementById("overlay-confidence");
const overlayChangeMask = document.getElementById("overlay-change-mask");
const svgOverlay = document.getElementById("svg-overlay");
const splitSlider = document.getElementById("splitSlider");
const sliderHandle = document.getElementById("sliderHandle");
const viewerWrapper = document.getElementById("viewerWrapper");
const tooltip = document.getElementById("tooltip");

// Checkbox Toggles
const toggleSliderMode = document.getElementById("toggleSliderMode");
const toggleBoundaries = document.getElementById("toggleBoundaries");
const toggleConfidence = document.getElementById("toggleConfidence");

// Info Elements
const infoUseCase = document.getElementById("infoUseCase");
const infoRegion = document.getElementById("infoRegion");
const infoBoundaries = document.getElementById("infoBoundaries");
const infoMappedArea = document.getElementById("infoMappedArea");
const infoGovRelevance = document.getElementById("infoGovRelevance");
const infoDescription = document.getElementById("infoDescription");

function init() {
  setupEventListeners();
  loadMode("agriculture");
}

function setupEventListeners() {
  // Mode Change
  modeSelect.addEventListener("change", (e) => {
    loadMode(e.target.value);
  });

  // Slider Mouse & Touch Events
  sliderHandle.addEventListener("mousedown", startDrag);
  window.addEventListener("mouseup", stopDrag);
  window.addEventListener("mousemove", onDrag);

  sliderHandle.addEventListener("touchstart", startDrag, { passive: true });
  window.addEventListener("touchend", stopDrag);
  window.addEventListener("touchmove", onTouchDrag, { passive: false });

  // Viewer Click to Move Slider
  viewerWrapper.addEventListener("click", (e) => {
    if (!toggleSliderMode.checked) return;
    const rect = viewerWrapper.getBoundingClientRect();
    const x = e.clientX - rect.left;
    updateSliderPosition((x / rect.width) * 100);
  });

  // Toggles
  toggleSliderMode.addEventListener("change", (e) => {
    if (e.target.checked) {
      sliderHandle.style.display = "block";
      updateSliderPosition(sliderPercent);
    } else {
      // Show full SR image
      sliderHandle.style.display = "none";
      imgSR.style.clipPath = "none";
    }
  });

  toggleBoundaries.addEventListener("change", (e) => {
    svgOverlay.style.display = e.target.checked ? "block" : "none";
  });

  toggleConfidence.addEventListener("change", (e) => {
    overlayConfidence.style.display = e.target.checked ? "block" : "none";
  });
}

function startDrag(e) {
  isDragging = true;
  e.preventDefault && e.preventDefault();
}

function stopDrag() {
  isDragging = false;
}

function onDrag(e) {
  if (!isDragging) return;
  const rect = viewerWrapper.getBoundingClientRect();
  const clientX = e.clientX;
  let percent = ((clientX - rect.left) / rect.width) * 100;
  percent = Math.max(0, Math.min(100, percent));
  updateSliderPosition(percent);
}

function onTouchDrag(e) {
  if (!isDragging) return;
  const rect = viewerWrapper.getBoundingClientRect();
  const touch = e.touches[0];
  let percent = ((touch.clientX - rect.left) / rect.width) * 100;
  percent = Math.max(0, Math.min(100, percent));
  updateSliderPosition(percent);
  e.preventDefault();
}

function updateSliderPosition(percent) {
  sliderPercent = percent;
  sliderHandle.style.left = `${percent}%`;
  imgSR.style.clipPath = `polygon(${percent}% 0, 100% 0, 100% 100%, ${percent}% 100%)`;
}

function loadMode(modeKey) {
  currentMode = modeKey;
  currentConfig = DEFAULT_CONFIGS[modeKey] || DEFAULT_CONFIGS.agriculture;

  // Try fetching external config file if running over http
  if (window.location.protocol.startsWith("http")) {
    fetch(`configs/${modeKey}.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) {
          currentConfig = { ...currentConfig, ...data };
          applyConfig(currentConfig);
        } else {
          applyConfig(currentConfig);
        }
      })
      .catch(() => applyConfig(currentConfig));
  } else {
    applyConfig(currentConfig);
  }
}

function applyConfig(cfg) {
  // Update Info Panel
  infoUseCase.textContent = cfg.title;
  infoRegion.textContent = cfg.region;
  infoBoundaries.textContent = `${cfg.boundaries_count} Delineated Vectors`;
  infoMappedArea.textContent = cfg.mapped_area;
  infoGovRelevance.textContent = cfg.gov_relevance;
  infoDescription.textContent = cfg.description || "Enhanced satellite mapping output.";

  // Update Images
  imgOriginal.src = cfg.original_image;
  imgSR.src = cfg.sr_image;
  overlayConfidence.src = cfg.confidence_overlay;

  if (cfg.change_mask) {
    overlayChangeMask.src = cfg.change_mask;
    overlayChangeMask.style.display = "block";
  } else {
    overlayChangeMask.style.display = "none";
  }

  // Render Boundaries
  renderGeoJSON(cfg.geojson_data);

  // Reset slider position
  updateSliderPosition(50);
}

function renderGeoJSON(geoData) {
  svgOverlay.innerHTML = "";
  if (!geoData || !geoData.features) return;

  const features = geoData.features;
  // Compute lon/lat bounding box
  let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
  features.forEach((feat) => {
    feat.geometry.coordinates[0].forEach(([lon, lat]) => {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    });
  });

  const padLon = (maxLon - minLon) * 0.15 || 0.002;
  const padLat = (maxLat - minLat) * 0.15 || 0.002;
  minLon -= padLon;
  maxLon += padLon;
  minLat -= padLat;
  maxLat += padLat;

  const width = 600;
  const height = 600;
  svgOverlay.setAttribute("viewBox", `0 0 ${width} ${height}`);

  function project(lon, lat) {
    const x = ((lon - minLon) / (maxLon - minLon)) * width;
    const y = ((maxLat - lat) / (maxLat - minLat)) * height;
    return [x, y];
  }

  features.forEach((feat, index) => {
    const coords = feat.geometry.coordinates[0];
    const pointsStr = coords
      .map(([lon, lat]) => {
        const [x, y] = project(lon, lat);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");

    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", pointsStr);
    polygon.setAttribute("class", "field-polygon");

    // Compute Centroid for Label
    let cx = 0, cy = 0;
    coords.forEach(([lon, lat]) => {
      const [x, y] = project(lon, lat);
      cx += x;
      cy += y;
    });
    cx /= coords.length;
    cy /= coords.length;

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", cx.toFixed(1));
    text.setAttribute("y", cy.toFixed(1));
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("class", "field-label");
    text.textContent = feat.properties.crop_type.split("/")[0].trim();

    // Tooltip Interactions
    polygon.addEventListener("mousemove", (e) => {
      const props = feat.properties;
      tooltip.style.display = "block";
      tooltip.style.left = `${e.pageX + 15}px`;
      tooltip.style.top = `${e.pageY + 15}px`;
      tooltip.innerHTML = `
        <h4>${props.parcel_id}</h4>
        <div><strong>Crop:</strong> ${props.crop_type}</div>
        <div><strong>Farmer:</strong> ${props.farmer_name}</div>
        <div><strong>Area:</strong> ${props.area_ha} Ha (${props.area_acres} Acres)</div>
        <div><strong>Perimeter:</strong> ${props.perimeter_m} m</div>
        <div style="color:#10b981; margin-top:4px;"><strong>PMFBY:</strong> Verified via DeepSRM</div>
      `;
    });

    polygon.addEventListener("mouseleave", () => {
      tooltip.style.display = "none";
    });

    svgOverlay.appendChild(polygon);
    svgOverlay.appendChild(text);
  });
}

document.addEventListener("DOMContentLoaded", init);
