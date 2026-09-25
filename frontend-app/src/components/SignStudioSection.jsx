import React, { useState } from 'react';
import { UploadCloud, CheckCircle2, Film, Sparkles, AlertCircle, ArrowUpRight, BarChart2 } from 'lucide-react';

export default function SignStudioSection() {
  const [dragActive, setDragActive] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processVideoFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      processVideoFile(e.target.files[0]);
    }
  };

  const processVideoFile = async (file) => {
    setAnalyzing(true);
    setError(null);
    setReport(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/studio/analyze-video', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setReport(data.report);
      } else {
        setError(data.error || 'Video analysis failed.');
      }
    } catch (err) {
      console.error('Video upload error:', err);
      setError('Connection to video studio service failed.');
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <section className="w-full max-w-5xl mx-auto px-4 py-8">
      {/* Deep Forest Emerald Showcase Container (Wispr Flow signature from Image 1) */}
      <div className="bg-[#0B352B] text-[#FAF8F2] rounded-[36px] sm:rounded-[44px] p-8 sm:p-12 shadow-2xl relative overflow-hidden">
        
        {/* Subtle Ambient Radial Light */}
        <div className="absolute -top-24 -right-24 w-96 h-96 rounded-full bg-emerald-400/10 blur-3xl pointer-events-none"></div>

        {/* Section Header */}
        <div className="max-w-2xl mb-8">
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-900/80 border border-emerald-700/50 text-[11px] font-semibold tracking-wider text-emerald-300 uppercase mb-3">
            <Film className="w-3 h-3" />
            <span>Sign Studio • Shazam for ASL</span>
          </div>

          <h2 className="text-4xl sm:text-5xl font-normal tracking-tight text-[#FAF8F2] leading-tight font-editorial">
            <span>4x faster </span>
            <span className="italic font-normal text-emerald-200">than typing.</span>
          </h2>
          <p className="text-sm sm:text-base text-emerald-100/80 mt-3 leading-relaxed">
            Upload any signing video. Lexis classifies 2,414 vocabulary signs with a 2-layer Bi-GRU neural network and generates natural English sentences with Gemini 2.5 Flash.
          </p>
        </div>

        {/* Wispr Flow Speed Comparison Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-8">
          <div className="bg-emerald-950/60 border border-emerald-800/40 rounded-3xl p-6 backdrop-blur-sm">
            <span className="text-xs uppercase tracking-wider text-emerald-400 font-medium block mb-1">
              Keyboard Typing
            </span>
            <div className="text-3xl sm:text-4xl font-normal font-editorial text-stone-300">
              45 <span className="text-lg font-sans text-stone-400">wpm</span>
            </div>
            <p className="text-xs text-emerald-200/60 mt-2">Traditional manual input</p>
          </div>

          <div className="bg-emerald-900/50 border border-emerald-500/30 rounded-3xl p-6 backdrop-blur-sm shadow-inner">
            <span className="text-xs uppercase tracking-wider text-emerald-300 font-semibold block mb-1">
              Lexis Continuous ASL
            </span>
            <div className="text-3xl sm:text-4xl font-normal font-editorial text-[#FAF8F2]">
              220 <span className="text-lg font-sans text-emerald-300">wpm</span>
            </div>
            <p className="text-xs text-emerald-200/80 mt-2">&lt;0.1ms vectorized continuous buffer</p>
          </div>
        </div>

        {/* Drag & Drop Video Upload Dropzone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-3xl p-8 sm:p-12 text-center transition-all cursor-pointer ${
            dragActive
              ? 'border-emerald-300 bg-emerald-900/60 scale-[1.01]'
              : 'border-emerald-700/60 hover:border-emerald-500 bg-emerald-950/40 hover:bg-emerald-950/60'
          }`}
        >
          <input
            type="file"
            accept="video/*"
            onChange={handleFileInput}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />

          <div className="flex flex-col items-center">
            <div className="w-16 h-16 rounded-2xl bg-emerald-900/80 border border-emerald-600/40 flex items-center justify-center mb-4 text-emerald-200 shadow-lg">
              <UploadCloud className="w-8 h-8" />
            </div>

            <p className="text-base font-semibold text-[#FAF8F2] mb-1">
              {analyzing ? 'Analyzing video keypoints...' : 'Drop signing video here, or browse'}
            </p>
            <p className="text-xs text-emerald-300/70 max-w-sm">
              Supports MP4, MOV, WebM. Identifies sign language and extracts Top-5 classifications across 2,414 classes.
            </p>
          </div>
        </div>

        {/* Error notification */}
        {error && (
          <div className="mt-4 p-4 rounded-2xl bg-rose-950/80 border border-rose-800 text-rose-200 text-xs text-center flex items-center justify-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Analysis Results Showcase Card */}
        {report && (
          <div className="mt-6 bg-[#07251E] rounded-3xl p-6 sm:p-8 border border-emerald-700/60 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 border-b border-emerald-800/60 pb-4 mb-6">
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                <span className="font-semibold text-base text-[#FAF8F2]">
                  {report.language_name}
                </span>
                <span className="text-xs font-mono bg-emerald-900 text-emerald-300 px-2 py-0.5 rounded-full">
                  {Math.round(report.language_confidence * 100)}% Confidence
                </span>
              </div>
              <span className="text-xs text-emerald-300/80">
                {report.total_frames_analyzed} frames processed
              </span>
            </div>

            {/* Natural English Translation from Gemini */}
            <div className="mb-6 bg-emerald-950/70 p-4 rounded-2xl border border-emerald-800/40">
              <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider block mb-1 flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-300" />
                Gemini 2.5 Flash English Translation:
              </span>
              <p className="text-xl font-normal font-editorial text-white">
                “{report.final_english_translation || 'No clear sentence formed.'}”
              </p>
            </div>

            {/* Top 5 Predicted Signs Breakdown */}
            {report.top5_predictions && report.top5_predictions.length > 0 && (
              <div>
                <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider block mb-3 flex items-center gap-1.5">
                  <BarChart2 className="w-3.5 h-3.5 text-emerald-400" />
                  Top Classifications (2,414 WLASL Classes):
                </span>
                <div className="space-y-2.5">
                  {report.top5_predictions.map((pred, i) => (
                    <div key={i} className="flex items-center justify-between text-xs">
                      <span className="font-medium text-emerald-100 capitalize w-28 truncate">
                        {pred.sign}
                      </span>
                      <div className="flex-1 mx-3 bg-emerald-950 rounded-full h-2 overflow-hidden border border-emerald-800/50">
                        <div
                          style={{ width: `${Math.round(pred.confidence * 100)}%` }}
                          className="bg-emerald-400 h-full rounded-full transition-all duration-500"
                        ></div>
                      </div>
                      <span className="font-mono text-emerald-300 w-12 text-right">
                        {Math.round(pred.confidence * 100)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
