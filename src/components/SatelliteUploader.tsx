import React, { useState, useRef } from "react";
import {
  UploadCloud,
  Cpu,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  FileImage,
  Layers,
  ArrowRight,
  RefreshCw,
  X,
  FileCode,
  Check
} from "lucide-react";
import {
  decodeSatelliteImage,
  runDeepSuperResolution,
  DecodedSatelliteImage,
  SRProcessingResult
} from "../srProcessor";

interface SatelliteUploaderProps {
  onProcessingComplete: (result: SRProcessingResult) => void;
  isOpen: boolean;
  onClose: () => void;
  activeModelKey?: string;
}

// Pre-packaged medium-resolution Sentinel-2 and Landsat benchmark datasets
const SAMPLE_PRESETS = [
  {
    id: "sentinel_krishna_geotiff",
    name: "Sentinel-2 4-Band GeoTIFF",
    sensor: "Sentinel-2 MSI",
    gsd: "10m GSD (GeoTIFF)",
    url: "/srm_prototype/data/original/sentinel2_krishna_4band.tif",
    description: "4-Band Multispectral GeoTIFF: Red, Green, Blue, NIR with uint16 DN"
  },
  {
    id: "sentinel_crop",
    name: "Sentinel-2 Andhra Fields",
    sensor: "Sentinel-2 MSI",
    gsd: "10m GSD (PNG)",
    url: "/srm_prototype/data/tiles/krishna_original.png",
    description: "Agricultural paddy parcels with irrigation canals"
  },
  {
    id: "sentinel_urban",
    name: "Sentinel-2 Urban Grid",
    sensor: "Sentinel-2 MSI",
    gsd: "10m GSD (PNG)",
    url: "/srm_prototype/data/tiles/guntur_urban_orig.png",
    description: "Medium-density urban building footprints & roads"
  },
  {
    id: "sentinel_flood",
    name: "Coastal Wetland & River",
    sensor: "Sentinel-2 MSI",
    gsd: "10m GSD (PNG)",
    url: "/srm_prototype/data/tiles/coastal_orig.png",
    description: "Water boundary and silt sediment plume"
  }
];

export const SatelliteUploader: React.FC<SatelliteUploaderProps> = ({
  onProcessingComplete,
  isOpen,
  onClose,
  activeModelKey
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedPresetUrl, setSelectedPresetUrl] = useState<string | null>(null);
  const [decodedData, setDecodedData] = useState<DecodedSatelliteImage | null>(null);
  const [selectedModel, setSelectedModel] = useState<"sen2sr" | "realesrgan">(
    activeModelKey === "realesrgan" ? "realesrgan" : "sen2sr"
  );
  const [scale, setScale] = useState<2 | 4>(4);
  const [isDecoding, setIsDecoding] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingStep, setProcessingStep] = useState<string>("");
  const [pipelineLogs, setPipelineLogs] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const appendLog = (msg: string) => {
    setPipelineLogs((prev) => [...prev.slice(-12), msg]);
    setProcessingStep(msg);
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processSelectedFile(e.target.files[0]);
    }
  };

  const processSelectedFile = async (file: File) => {
    setErrorMsg(null);
    setSelectedFile(file);
    setSelectedPresetUrl(null);
    setIsDecoding(true);
    setPipelineLogs([]);

    try {
      appendLog(`[TIFF Decoder] Reading uploaded file: ${file.name} (${Math.round(file.size / 1024)} KB)...`);
      const decoded = await decodeSatelliteImage(file, appendLog);
      setDecodedData(decoded);
    } catch (err: any) {
      console.error("TIFF Decoding error:", err);
      setErrorMsg(`Failed to decode satellite image: ${err.message || "Invalid raster format"}`);
    } finally {
      setIsDecoding(false);
    }
  };

  const handleSelectSample = async (sampleUrl: string) => {
    setErrorMsg(null);
    setSelectedFile(null);
    setSelectedPresetUrl(sampleUrl);
    setIsDecoding(true);
    setPipelineLogs([]);

    try {
      appendLog(`[TIFF Decoder] Fetching preset raster: ${sampleUrl.split("/").pop()}...`);
      const decoded = await decodeSatelliteImage(sampleUrl, appendLog);
      setDecodedData(decoded);
    } catch (err: any) {
      console.error("Preset Loading error:", err);
      setErrorMsg(`Failed to load preset: ${err.message || "Network error"}`);
    } finally {
      setIsDecoding(false);
    }
  };

  const handleExecuteSuperResolution = async () => {
    if (!decodedData) {
      setErrorMsg("Please upload a satellite image or select a benchmark scene first.");
      return;
    }

    try {
      setIsProcessing(true);
      setErrorMsg(null);

      appendLog(`[Super-Resolution Pipeline] Starting ${scale}x deep-learning enhancement on ${decodedData.sourceName}...`);
      const result = await runDeepSuperResolution(decodedData, selectedModel, scale, appendLog);

      appendLog("[Pipeline Complete] Output ready for geospatial inspection & download.");
      await new Promise((r) => setTimeout(r, 200));

      onProcessingComplete(result);
      onClose();
    } catch (err: any) {
      console.error("Super-resolution pipeline error:", err);
      setErrorMsg(err.message || "Deep learning super-resolution inference failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md animate-fadeIn">
      <div className="bg-[#111827] border border-[#232D42] rounded-2xl w-full max-w-2xl overflow-hidden shadow-2xl flex flex-col max-h-[92vh]">
        
        {/* Header */}
        <div className="px-6 py-4 bg-[#151C2C] border-b border-[#232D42] flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
              <UploadCloud className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                Upload Medium-Resolution Satellite Image
              </h2>
              <p className="text-xs text-slate-400">
                Enhance 10m Sentinel-2 (GeoTIFF / PNG) to 2.5m Ground Sampling Distance with Sen2SR
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isProcessing}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1 custom-scrollbar">
          
          {/* Upload Drop Zone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200 text-center ${
              dragActive
                ? "border-emerald-500 bg-emerald-500/10 scale-[1.01]"
                : decodedData
                ? "border-emerald-500/40 bg-[#0B0F1A]"
                : "border-[#2A364F] hover:border-emerald-500/50 bg-[#0B0F1A]/60"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff,.geotiff,image/tiff,image/png,image/jpeg,image/webp"
              onChange={handleFileChange}
              className="hidden"
            />

            {isDecoding ? (
              <div className="flex flex-col items-center gap-2 py-4">
                <RefreshCw className="w-8 h-8 text-emerald-400 animate-spin" />
                <span className="text-xs text-emerald-300 font-semibold">Reading GeoTIFF bands & metadata...</span>
              </div>
            ) : decodedData ? (
              <div className="flex flex-col items-center gap-3 w-full">
                <div className="relative w-44 h-44 rounded-xl overflow-hidden border border-[#232D42] shadow-inner bg-black">
                  <img
                    src={decodedData.previewDataUrl}
                    alt="Uploaded Satellite Preview"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-2 right-2 px-2 py-0.5 rounded bg-black/85 text-[9px] text-emerald-400 font-bold tracking-wider uppercase border border-emerald-500/40">
                    {decodedData.isGeoTiff ? "GeoTIFF" : "RGB Image"}
                  </div>
                  <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded bg-black/85 text-[9px] text-slate-300 font-mono">
                    {decodedData.width} × {decodedData.height} px
                  </div>
                </div>

                <div className="flex flex-col items-center gap-1">
                  <div className="flex items-center gap-2 text-xs text-emerald-400 font-semibold">
                    <CheckCircle2 className="w-4 h-4 shrink-0" />
                    <span className="truncate max-w-[320px]">{decodedData.sourceName}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
                    <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
                      {decodedData.bands} Band{decodedData.bands > 1 ? "s" : ""}
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
                      {decodedData.bitsPerSample}-bit
                    </span>
                    <span className="px-1.5 py-0.5 rounded bg-emerald-950/70 border border-emerald-500/30 text-emerald-300">
                      10m Sentinel-2 GSD
                    </span>
                  </div>
                </div>

                <span className="text-[11px] text-slate-400">Click or drag another .tif / image to replace</span>
              </div>
            ) : (
              <>
                <div className="w-12 h-12 rounded-full bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
                  <FileImage className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-200">
                    Click to browse or drop Sentinel-2 GeoTIFF (.tif / .tiff) here
                  </p>
                  <p className="text-xs text-slate-400 mt-1">
                    Accepts Single-band, 3-Band RGB, and 4-Band (RGB+NIR) uint8/uint16 GeoTIFFs & PNGs
                  </p>
                </div>
              </>
            )}
          </div>

          {/* Quick Preset Samples */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1">
                <FileCode className="w-3.5 h-3.5 text-emerald-400" />
                Or choose a benchmark test scene:
              </span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {SAMPLE_PRESETS.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => handleSelectSample(sample.url)}
                  className={`p-2 rounded-xl border text-left transition cursor-pointer flex flex-col gap-1 ${
                    selectedPresetUrl === sample.url
                      ? "bg-emerald-500/15 border-emerald-500/50 text-white"
                      : "bg-[#151C2C] border-[#232D42] text-slate-400 hover:text-slate-200 hover:border-slate-600"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-200 truncate">{sample.name.split(" ")[0]}</span>
                    {selectedPresetUrl === sample.url && <Check className="w-3 h-3 text-emerald-400 shrink-0" />}
                  </div>
                  <span className="text-[9px] text-emerald-400 font-mono truncate">{sample.gsd}</span>
                  <span className="text-[9px] text-slate-400 line-clamp-1">{sample.description}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Deep Learning Model Selection */}
          <div className="space-y-2">
            <span className="text-[11px] font-bold uppercase tracking-widest text-slate-400 flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-emerald-400" />
              Choose Deep Learning Architecture
            </span>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Option 1: Sen2SR */}
              <div
                onClick={() => setSelectedModel("sen2sr")}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                  selectedModel === "sen2sr"
                    ? "bg-emerald-500/10 border-emerald-500/60 text-white shadow-lg shadow-emerald-950/40"
                    : "bg-[#151C2C] border-[#232D42] text-slate-400 hover:text-slate-200"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm font-bold text-emerald-300 flex items-center gap-1.5">
                      <Sparkles className="w-4 h-4 text-emerald-400" />
                      Sen2SR (ESA OpenSR)
                    </span>
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                      2.5m GSD
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-snug">
                    Designed specifically for Sentinel-2 multi-spectral data. Preserves physical surface reflectance with strict radiometric conservation.
                  </p>
                </div>
                <div className="mt-3 pt-2 border-t border-[#232D42] flex items-center justify-between text-[10px] text-emerald-400/90 font-mono">
                  <span>RCAB + Hard Constraint</span>
                  <span className="font-bold">ΔE = 0.00%</span>
                </div>
              </div>

              {/* Option 2: Real-ESRGAN */}
              <div
                onClick={() => setSelectedModel("realesrgan")}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all flex flex-col justify-between ${
                  selectedModel === "realesrgan"
                    ? "bg-emerald-500/10 border-emerald-500/60 text-white shadow-lg shadow-emerald-950/40"
                    : "bg-[#151C2C] border-[#232D42] text-slate-400 hover:text-slate-200"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-sm font-bold text-slate-100 flex items-center gap-1.5">
                      <Layers className="w-4 h-4 text-blue-400" />
                      Real-ESRGAN (RRDBNet)
                    </span>
                    <span className="text-[9px] font-mono px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-300 border border-blue-500/30">
                      Edge Sharpener
                    </span>
                  </div>
                  <p className="text-xs text-slate-300 leading-snug">
                    Deep residual dense network with perceptual gradients for crisp building footprints and road infrastructure.
                  </p>
                </div>
                <div className="mt-3 pt-2 border-t border-[#232D42] flex items-center justify-between text-[10px] text-slate-400 font-mono">
                  <span>Residual Dense Blocks</span>
                  <span className="font-bold">Edge Focus</span>
                </div>
              </div>
            </div>
          </div>

          {/* Scale Selector */}
          <div className="flex items-center justify-between bg-[#151C2C] p-3 rounded-xl border border-[#232D42]">
            <div>
              <span className="text-xs font-bold text-slate-200">Super-Resolution Scale Factor</span>
              <p className="text-[11px] text-slate-400">
                {scale === 4 ? "4x Super-Resolution (10m → 2.5m Ground Sampling Distance)" : "2x Super-Resolution (10m → 5.0m GSD)"}
              </p>
            </div>
            <div className="flex items-center gap-1 bg-[#0B0F1A] p-1 rounded-lg border border-[#232D42]">
              <button
                type="button"
                onClick={() => setScale(2)}
                className={`px-3 py-1 rounded text-xs font-bold transition cursor-pointer ${
                  scale === 2 ? "bg-emerald-500 text-black shadow" : "text-slate-400 hover:text-white"
                }`}
              >
                2x (5m)
              </button>
              <button
                type="button"
                onClick={() => setScale(4)}
                className={`px-3 py-1 rounded text-xs font-bold transition cursor-pointer ${
                  scale === 4 ? "bg-emerald-500 text-black shadow" : "text-slate-400 hover:text-white"
                }`}
              >
                4x (2.5m)
              </button>
            </div>
          </div>

          {/* Processing Status & Diagnostics Log Stream */}
          {isProcessing && (
            <div className="p-3.5 bg-emerald-950/40 border border-emerald-500/40 rounded-xl space-y-2">
              <div className="flex items-center gap-2.5">
                <RefreshCw className="w-4 h-4 text-emerald-400 animate-spin shrink-0" />
                <span className="text-xs font-bold text-emerald-300">
                  Executing Deep Neural Network Forward Pass ({selectedModel.toUpperCase()})...
                </span>
              </div>
              <div className="bg-[#0B0F1A] p-2 rounded-lg border border-[#232D42] max-h-24 overflow-y-auto font-mono text-[10px] text-emerald-400/90 space-y-1">
                {pipelineLogs.map((l, i) => (
                  <div key={i} className="truncate leading-tight">
                    {l}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error Message */}
          {errorMsg && (
            <div className="p-3 bg-red-950/40 border border-red-500/50 rounded-xl flex items-center gap-2.5 text-red-300 text-xs">
              <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 bg-[#151C2C] border-t border-[#232D42] flex items-center justify-between">
          <button
            type="button"
            onClick={onClose}
            disabled={isProcessing}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-400 hover:text-white hover:bg-slate-800 transition cursor-pointer"
          >
            Cancel
          </button>

          <button
            type="button"
            onClick={handleExecuteSuperResolution}
            disabled={!decodedData || isProcessing}
            className={`px-5 py-2 rounded-xl text-xs font-bold flex items-center gap-2 transition-all cursor-pointer shadow-lg ${
              !decodedData || isProcessing
                ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700"
                : "bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold shadow-emerald-500/20 scale-[1.02]"
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running Inference ({scale}x)...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                <span>Run Super-Resolution ({selectedModel === "sen2sr" ? "Sen2SR 2.5m" : "Real-ESRGAN"})</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>

      </div>
    </div>
  );
};
