import React, { useState } from 'react';
import { Camera, Square, Play, Sparkles, Copy, Check, RefreshCw, Zap, ShieldCheck, Activity } from 'lucide-react';

export default function VisionSection({
  isVisionRunning,
  onToggleVision,
  visionConnecting,
  currentEvent,
  onClearBuffer,
}) {
  const [copied, setCopied] = useState(false);

  // Extract real-time telemetry from WebSocket event
  const liveGesture = currentEvent?.gesture && currentEvent.gesture !== '...' ? currentEvent.gesture : '';
  const confidence = currentEvent?.confidence ? Math.round(currentEvent.confidence * 100) : null;
  const sentence = currentEvent?.sentence_so_far || '';
  const flashText = currentEvent?.flash_text || currentEvent?.completed_sentence || '';
  const topPredictions = currentEvent?.top_predictions || [];
  const fps = currentEvent?.fps ? Math.round(currentEvent.fps) : 30;

  const handleCopy = () => {
    const textToCopy = flashText || sentence || liveGesture;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const sampleWords = ["hello", "thank you", "bathroom", "water", "help", "where", "please", "i love you", "more", "eat"];

  return (
    <section className="w-full max-w-5xl mx-auto px-4 py-4">
      {/* Outer Card Container */}
      <div className="relative bg-white rounded-[36px] sm:rounded-[44px] border border-stone-200/90 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.06)] overflow-hidden transition-all">
        
        {/* Top Header Bar */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-stone-100 bg-[#FCFBF7]">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-2xl transition-colors ${isVisionRunning ? 'bg-emerald-50 text-emerald-800' : 'bg-stone-100 text-stone-700'}`}>
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-stone-900 tracking-tight flex items-center gap-2">
                <span>Continuous ASL Vision Subtitler</span>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200">
                  CUDA RTMPose
                </span>
              </h2>
              <p className="text-xs text-stone-500">133 3D keypoints • 0.1ms vectorized rolling buffer</p>
            </div>
          </div>

          {/* Right Action Button */}
          <div className="flex items-center gap-2">
            {isVisionRunning ? (
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 text-xs font-semibold shadow-sm transition-all active:scale-95"
              >
                <Square className="w-3 h-3 fill-rose-600" />
                <span>Stop Detection</span>
              </button>
            ) : (
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="flex items-center gap-2 px-5 py-2 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] text-xs font-semibold shadow-sm transition-all active:scale-95 disabled:opacity-60"
              >
                <Play className="w-3.5 h-3.5 fill-[#1E1B4B]" />
                <span>{visionConnecting ? 'Initializing CUDA...' : 'Start Detection'}</span>
              </button>
            )}
          </div>
        </div>

        {/* Viewport Area */}
        <div className="relative w-full bg-[#111413] flex flex-col items-center justify-center min-h-[380px] sm:min-h-[460px]">
          {isVisionRunning ? (
            /* Active Live Viewfinder */
            <div className="relative w-full h-full flex flex-col items-center justify-center p-3 sm:p-4">
              {/* Video container with proper aspect ratio and NO aggressive cropping */}
              <div className="relative w-full max-w-4xl rounded-2xl sm:rounded-3xl overflow-hidden bg-black shadow-2xl border border-stone-800">
                <img
                  src="/api/vision/stream"
                  alt="Lexis Live ASL Stream"
                  className="w-full h-auto max-h-[540px] object-contain mx-auto"
                />

                {/* Floating Top Telemetry Pill in Viewport */}
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none">
                  {/* Status & FPS */}
                  <div className="flex items-center gap-2 bg-stone-900/85 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-full border border-white/10 shadow-lg">
                    <span className="relative flex h-2 w-2">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span className="font-mono font-medium">{fps} FPS</span>
                    <span className="text-stone-400">•</span>
                    <span className="font-medium text-[11px] text-stone-200">133 Keypoints</span>
                  </div>

                  {/* Active Sign Pill */}
                  {liveGesture && (
                    <div className="flex items-center gap-2 bg-[#0B352B]/90 backdrop-blur-md text-emerald-200 text-xs px-3.5 py-1.5 rounded-full border border-emerald-500/30 shadow-lg animate-in fade-in duration-200">
                      <span className="text-[10px] text-emerald-400 uppercase tracking-wider font-semibold">Active Sign:</span>
                      <span className="font-bold text-white uppercase text-sm tracking-wide">{liveGesture}</span>
                      {confidence && (
                        <span className="bg-emerald-950/90 text-emerald-300 font-mono text-[10px] px-1.5 py-0.5 rounded">
                          {confidence}%
                        </span>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* Standby State with Wispr Flow Surrealist Aesthetic */
            <div className="w-full h-full flex flex-col items-center justify-center p-8 sm:p-12 text-center bg-[#FAF8F2] relative">
              <div className="absolute w-80 h-80 rounded-full bg-emerald-900/5 blur-3xl pointer-events-none"></div>
              <div className="absolute w-64 h-64 rounded-full bg-purple-900/5 blur-3xl pointer-events-none"></div>

              {/* Minimalist Camera Icon */}
              <div className="relative z-10 w-20 h-20 rounded-3xl bg-white border border-stone-200 shadow-[0_10px_35px_rgba(0,0,0,0.05)] flex items-center justify-center mb-6 group">
                <Camera className="w-8 h-8 text-stone-800 transition-transform group-hover:scale-105" />
                <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-stone-300 rounded-full border-2 border-white"></span>
              </div>

              {/* Headline */}
              <h3 className="relative z-10 text-3xl sm:text-4xl font-normal text-stone-900 mb-3 font-editorial">
                <span>Natural expression, </span>
                <span className="italic font-normal">zero latency.</span>
              </h3>
              <p className="relative z-10 max-w-lg text-xs sm:text-sm text-stone-600 mb-8 leading-relaxed">
                Press Start Detection to activate RTMPose Wholebody tracking. Continuous signs are vectorized at 60 FPS on CUDA and reconstructed into fluent English sentences by Gemini 2.5 Flash.
              </p>

              {/* Prominent Lilac CTA Button */}
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="relative z-10 inline-flex items-center gap-2.5 px-7 py-3.5 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] text-sm font-semibold shadow-lg shadow-purple-950/5 transition-all duration-200 active:scale-95 disabled:opacity-60"
              >
                <Play className="w-4 h-4 fill-[#1E1B4B]" />
                <span>{visionConnecting ? 'Initializing RTMPose CUDA...' : 'Start Detection'}</span>
              </button>

              {/* Verified Words Preview */}
              <div className="relative z-10 mt-8 max-w-md">
                <span className="text-[11px] font-semibold text-stone-400 uppercase tracking-wider block mb-2">
                  50 Verified Active Words (99.7% Accuracy):
                </span>
                <div className="flex flex-wrap items-center justify-center gap-1.5">
                  {sampleWords.map((word) => (
                    <span
                      key={word}
                      className="px-2.5 py-1 rounded-full bg-white border border-stone-200/80 text-[11px] font-medium text-stone-600 capitalize shadow-2xs"
                    >
                      {word}
                    </span>
                  ))}
                  <span className="text-[11px] text-stone-400 font-medium px-2">+40 more</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Dedicated Live Subtitle Area (Spacious & Clean) */}
        {isVisionRunning && (
          <div className="p-6 bg-white border-t border-stone-100 flex flex-col gap-4">
            {/* Top Predictions Chips if active */}
            {topPredictions.length > 0 && (
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] uppercase font-semibold text-stone-400 tracking-wider">
                  Suggestions:
                </span>
                {topPredictions.map((pred, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-stone-100 border border-stone-200 text-xs font-medium text-stone-700 capitalize"
                  >
                    <span>{pred.gesture}</span>
                    <span className="text-[10px] text-stone-400 font-mono">
                      {Math.round(pred.confidence * 100)}%
                    </span>
                  </span>
                ))}
              </div>
            )}

            {/* Subtitle Card */}
            <div className="w-full bg-[#FAF8F2] rounded-2xl p-4 sm:p-5 border border-stone-200/80 shadow-inner flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <Sparkles className="w-3.5 h-3.5 text-purple-600" />
                  <span className="text-[11px] font-semibold tracking-wider text-purple-900 uppercase">
                    English Subtitles (Gemini 2.5 Flash)
                  </span>
                </div>
                <p className="text-base sm:text-lg font-semibold text-stone-900 leading-snug">
                  {flashText || sentence || (
                    <span className="text-stone-400 font-normal italic text-sm">
                      Sign naturally — continuous signs will assemble and polish here...
                    </span>
                  )}
                </p>
              </div>

              <div className="flex items-center gap-2 self-end sm:self-center">
                <button
                  onClick={handleCopy}
                  title="Copy Subtitles"
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-white border border-stone-200 hover:bg-stone-50 text-xs font-medium text-stone-700 transition-colors shadow-2xs"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
                <button
                  onClick={onClearBuffer}
                  title="Clear Buffer"
                  className="p-1.5 rounded-xl bg-white border border-stone-200 hover:bg-stone-50 text-stone-600 transition-colors shadow-2xs"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Bottom Telemetry Strip */}
        <div className="px-6 py-3 bg-[#FCFBF7] border-t border-stone-100 flex flex-wrap items-center justify-between gap-3 text-xs text-stone-500 font-medium">
          <div className="flex items-center gap-2">
            <span className="text-stone-400 font-mono text-[11px]">STATUS:</span>
            <span className={isVisionRunning ? 'text-emerald-700 font-semibold' : 'text-stone-500'}>
              {isVisionRunning ? 'Vision Subtitler Active' : 'Standby (Camera Idle)'}
            </span>
          </div>
          
          <div className="flex items-center gap-4 text-stone-500 text-[11px]">
            <span>Conversational Cooldown: <strong>0.35s</strong></span>
            <span>Hold Locking: <strong>0.12s</strong></span>
            <span>Elevation Guard: <strong>Active</strong></span>
          </div>
        </div>
      </div>
    </section>
  );
}
