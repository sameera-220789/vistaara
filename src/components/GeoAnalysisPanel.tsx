import React from "react";
import {
  ImageAnalysisResult,
  generateDynamicGeoJson,
  generateDynamicCsv
} from "../analysisEngine";
import {
  Download,
  AlertTriangle,
  CheckCircle2,
  Layers,
  Activity,
  BarChart3,
  Globe,
  Trees,
  Droplets,
  Building2,
  Maximize2
} from "lucide-react";

interface GeoAnalysisPanelProps {
  analysis: ImageAnalysisResult | null;
  isAnalyzing?: boolean;
  onDownloadRaster: () => void;
}

export const GeoAnalysisPanel: React.FC<GeoAnalysisPanelProps> = ({
  analysis,
  isAnalyzing = false,
  onDownloadRaster
}) => {
  if (isAnalyzing) {
    return (
      <div className="bg-[#111827] border border-[#232D42] rounded-2xl p-6 flex flex-col items-center justify-center min-h-[300px] text-center shadow-lg">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin mb-3" />
        <p className="text-xs font-bold text-white uppercase tracking-wider">
          Computing Pixel-Level Geo-Analysis...
        </p>
        <p className="text-[11px] text-slate-400 mt-1">
          Calculating dynamic NDVI, land-cover segments & spectral entropy.
        </p>
      </div>
    );
  }

  if (!analysis) {
    return (
      <div className="bg-[#111827] border border-[#232D42] rounded-2xl p-6 flex flex-col items-center justify-center min-h-[260px] text-center shadow-lg">
        <p className="text-xs text-slate-400">
          Upload a satellite image or select a benchmark scene to view dynamic analysis.
        </p>
      </div>
    );
  }

  const handleExportGeoJson = () => {
    const geoJsonStr = generateDynamicGeoJson(analysis);
    const blob = new Blob([geoJsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deep_srm_analysis_${analysis.imageHash}.geojson`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleExportCsv = () => {
    const csvStr = generateDynamicCsv(analysis);
    const blob = new Blob([csvStr], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `deep_srm_analysis_${analysis.imageHash}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const veg = analysis.vegetation;
  const lc = analysis.landCover;
  const stress = analysis.stress;
  const stats = analysis.statistics;

  return (
    <div className="bg-[#111827] border border-[#232D42] rounded-2xl p-5 flex flex-col shadow-lg gap-4 text-xs">
      
      {/* Header with Image Identifier Hash */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#232D42] pb-3">
        <div>
          <div className="flex items-center gap-1.5">
            <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
            <h3 className="text-[11px] font-bold text-white uppercase tracking-wider">
              Real Image-Dependent Geo-Analysis
            </h3>
          </div>
          <p className="text-[10px] text-slate-400 font-mono mt-0.5 truncate max-w-[280px]">
            ID: {analysis.imageHash}
          </p>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold">
            {analysis.numBands} Band(s)
          </span>
          <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-[#151C2C] border border-[#232D42] text-slate-300">
            {analysis.width}×{analysis.height}
          </span>
        </div>
      </div>

      {/* 1. Vegetation Analysis Section */}
      <div className="bg-[#0B0F1A] border border-[#232D42] rounded-xl p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Trees className="w-3 h-3 text-emerald-400" />
            Vegetation Index
          </span>
          {veg.hasRequiredBandsForNdvi ? (
            <span className="text-[10px] font-mono font-bold text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-500/40">
              Mean NDVI: {veg.meanNdvi}
            </span>
          ) : (
            <span className="text-[9px] font-mono font-bold text-amber-400 bg-amber-950/80 px-2 py-0.5 rounded border border-amber-500/40">
              GLI: {veg.visibleIndexMean} (Visible)
            </span>
          )}
        </div>

        {/* Band Validation Notice */}
        {!veg.hasRequiredBandsForNdvi && (
          <div className="p-2.5 bg-amber-950/40 border border-amber-500/30 rounded-lg text-amber-300 flex items-start gap-2">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
            <div className="text-[10px] leading-relaxed">
              <strong className="block text-amber-200">This analysis requires Red + NIR bands</strong>
              Scene has {analysis.numBands} visible band(s). NDVI calculation was bypassed to prevent inaccurate estimation. Calculated Visible Green Leaf Index (GLI) across RGB channels.
            </div>
          </div>
        )}

        {/* Dynamic Vegetation Breakdown */}
        <div className="grid grid-cols-2 gap-2 text-[11px]">
          <div className="bg-[#151C2C] p-2 rounded-lg border border-[#232D42]">
            <span className="text-slate-400 text-[10px] block">Vegetation Coverage</span>
            <span className="font-bold text-emerald-400 font-mono text-sm">
              {veg.vegetationCoveragePct}%
            </span>
          </div>
          <div className="bg-[#151C2C] p-2 rounded-lg border border-[#232D42]">
            <span className="text-slate-400 text-[10px] block">High Density Canopy</span>
            <span className="font-bold text-white font-mono text-sm">
              {veg.classification.highVegetationPct}%
            </span>
          </div>
        </div>

        {/* Vegetation Density Tiers */}
        <div className="space-y-1 pt-1 text-[10px]">
          <div className="flex justify-between text-slate-400">
            <span>Moderate Density:</span>
            <span className="font-mono text-slate-200">{veg.classification.moderateVegetationPct}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Sparse / Low Density:</span>
            <span className="font-mono text-slate-200">{veg.classification.lowVegetationPct}%</span>
          </div>
          <div className="flex justify-between text-slate-400">
            <span>Non-Vegetated Area:</span>
            <span className="font-mono text-slate-200">{veg.classification.nonVegetationPct}%</span>
          </div>
        </div>
      </div>

      {/* 2. Land-Cover & Terrain Distribution */}
      <div className="bg-[#0B0F1A] border border-[#232D42] rounded-xl p-3.5 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Layers className="w-3 h-3 text-cyan-400" />
            Land-Cover Classification
          </span>
          <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-cyan-950/80 border border-cyan-500/40 text-cyan-300">
            {lc.dominantClass}
          </span>
        </div>

        {/* Dynamic Horizontal Stacked Progress Bar */}
        <div className="w-full h-3 rounded-full overflow-hidden flex bg-[#151C2C] border border-[#232D42]">
          <div
            style={{ width: `${lc.vegetationPct}%` }}
            className="bg-emerald-500 h-full transition-all duration-300"
            title={`Vegetation: ${lc.vegetationPct}%`}
          />
          <div
            style={{ width: `${lc.waterPct}%` }}
            className="bg-sky-500 h-full transition-all duration-300"
            title={`Water: ${lc.waterPct}%`}
          />
          <div
            style={{ width: `${lc.bareSoilPct}%` }}
            className="bg-amber-500 h-full transition-all duration-300"
            title={`Soil: ${lc.bareSoilPct}%`}
          />
          <div
            style={{ width: `${lc.builtUpPct}%` }}
            className="bg-fuchsia-500 h-full transition-all duration-300"
            title={`Built-Up: ${lc.builtUpPct}%`}
          />
          <div
            style={{ width: `${lc.otherPct}%` }}
            className="bg-slate-500 h-full transition-all duration-300"
            title={`Other: ${lc.otherPct}%`}
          />
        </div>

        {/* Land Cover Metrics List */}
        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px] pt-1">
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-slate-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />
              Vegetation:
            </span>
            <span className="font-mono text-emerald-400 font-bold">{lc.vegetationPct}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-slate-400">
              <span className="w-2 h-2 rounded-full bg-sky-500 inline-block" />
              Water Bodies:
            </span>
            <span className="font-mono text-sky-400 font-bold">{lc.waterPct}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-slate-400">
              <span className="w-2 h-2 rounded-full bg-amber-500 inline-block" />
              Bare Soil:
            </span>
            <span className="font-mono text-amber-400 font-bold">{lc.bareSoilPct}%</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="flex items-center gap-1 text-slate-400">
              <span className="w-2 h-2 rounded-full bg-fuchsia-500 inline-block" />
              Built-Up:
            </span>
            <span className="font-mono text-fuchsia-400 font-bold">{lc.builtUpPct}%</span>
          </div>
        </div>
      </div>

      {/* 3. Vegetation Condition & Stressed Regions */}
      <div className="bg-[#0B0F1A] border border-[#232D42] rounded-xl p-3.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1">
            <Activity className="w-3 h-3 text-rose-400" />
            Vegetative Condition
          </span>
          <span className="text-[9px] font-bold px-2 py-0.5 rounded bg-rose-950/80 border border-rose-500/40 text-rose-300">
            {stress.severityLabel}
          </span>
        </div>
        <p className="text-[10px] text-slate-300 leading-snug">
          {stress.anomalyDescription}
        </p>
        <div className="flex justify-between items-center text-[10px] pt-1 text-slate-400 border-t border-[#1C2538]">
          <span>Stressed Canopy Ratio:</span>
          <span className="font-mono font-bold text-rose-400">
            {stress.stressedVegetationPct}% of vegetation
          </span>
        </div>
      </div>

      {/* 4. Radiometric & Information Theory Statistics */}
      <div className="bg-[#0B0F1A] border border-[#232D42] rounded-xl p-3 space-y-1.5 text-[10px]">
        <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mb-1">
          Dynamic Pixel Statistics
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-slate-300 font-mono">
          <div className="flex justify-between">
            <span className="text-slate-500">Min Pixel:</span>
            <span>{stats.minPixelValue}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Max Pixel:</span>
            <span>{stats.maxPixelValue}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Mean Lum:</span>
            <span>{stats.meanPixelValue}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Std Dev:</span>
            <span>{stats.stdDev}</span>
          </div>
          <div className="flex justify-between col-span-2 pt-1 border-t border-[#1C2538]">
            <span className="text-slate-500">Shannon Entropy:</span>
            <span className="text-emerald-400 font-bold">{stats.shannonEntropy} bits/px</span>
          </div>
        </div>
      </div>

      {/* 5. Export Actions */}
      <div className="pt-2 border-t border-[#232D42] flex flex-col gap-2">
        <button
          onClick={onDownloadRaster}
          className="w-full flex items-center justify-center gap-1.5 py-2 px-2.5 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 text-emerald-400 text-xs font-bold transition-all cursor-pointer shadow-sm"
        >
          <Download className="w-3.5 h-3.5" />
          <span>Download Super-Resolved 2.5m Raster</span>
        </button>
        <div className="flex items-center gap-2">
          <button
            onClick={handleExportGeoJson}
            className="flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-lg bg-[#151C2C] hover:bg-[#1f293d] border border-[#232D42] text-emerald-400 text-[11px] font-semibold transition-all cursor-pointer"
          >
            <Download className="w-3 h-3" />
            <span>GeoJSON</span>
          </button>
          <button
            onClick={handleExportCsv}
            className="flex-1 flex items-center justify-center gap-1.5 py-1.5 px-2.5 rounded-lg bg-[#151C2C] hover:bg-[#1f293d] border border-[#232D42] text-slate-300 text-[11px] font-semibold transition-all cursor-pointer"
          >
            <Download className="w-3 h-3" />
            <span>CSV Summary</span>
          </button>
        </div>
      </div>

    </div>
  );
};
