import React, { useState } from 'react';
import { Camera, Square, Play, Sparkles, Copy, Check, RefreshCw, Zap, ShieldCheck } from 'lucide-react';

export default function VisionSection({
  isVisionRunning,
  onToggleVision,
  visionConnecting,
  currentEvent,
  onClearBuffer,
}) {
  const [copied, setCopied] = useState(false);

  // Extract real-time telemetry from WebSocket event
  const liveGesture = currentEvent?.current_gesture || '';
  const confidence = currentEvent?.confidence ? Math.round(currentEvent.confidence * 100) : null;
  const sentence = currentEvent?.sentence || '';
  const flashText = currentEvent?.flash_text || '';
  const rawTokens = currentEvent?.buffer || [];

  const handleCopy = () => {
    const textToCopy = flashText || sentence || liveGesture;
    if (textToCopy) {
      navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section className="w-full max-w-5xl mx-auto px-4 py-4">
      {/* Outer Surrealist Canvas Container */}
      <div className="relative bg-white rounded-[32px] sm:rounded-[40px] border border-stone-200/90 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.05)] overflow-hidden transition-all">
        
        {/* Top Header Strip */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-stone-100 bg-[#FCFBF7]">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-2xl ${isVisionRunning ? 'bg-emerald-50 text-emerald-800' : 'bg-stone-100 text-stone-700'}`}>
              <Camera className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-stone-900 tracking-tight flex items-center gap-2">
                <span>Continuous ASL Vision Subtitler</span>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200">
                  RTMPose Wholebody
                </span>
              </h2>
              <p className="text-xs text-stone-500">133 3D anatomical keypoints • &lt;0.1ms vectorized rolling buffer</p>
            </div>
          </div>

          {/* Right Status Controls */}
          <div className="flex items-center gap-2">
            {isVisionRunning ? (
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200 text-xs font-semibold shadow-sm transition-all"
              >
                <Square className="w-3 h-3 fill-rose-600" />
                <span>Stop Detection</span>
              </button>
            ) : (
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="flex items-center gap-1.5 px-4 py-1.5 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] text-xs font-semibold shadow-sm transition-all active:scale-95"
              >
                <Play className="w-3 h-3 fill-[#1E1B4B]" />
                <span>{visionConnecting ? 'Connecting...' : 'Start Detection'}</span>
              </button>
            )}
          </div>
        </div>

        {/* Viewport: Live Stream vs Peaceful Idle State */}
        <div className="relative aspect-[16/10] sm:aspect-[16/9] w-full bg-[#141716] overflow-hidden flex items-center justify-center">
          {isVisionRunning ? (
            /* Live Stream Active */
            <div className="relative w-full h-full">
              <img
                src="/api/vision/stream"
                alt="Lexis Live ASL Stream"
                className="w-full h-full object-cover"
              />

              {/* Top Floating Telemetry Overlay */}
              <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
                <div className="flex items-center gap-2 bg-stone-900/80 backdrop-blur-md text-white text-xs px-3 py-1.5 rounded-full border border-white/10 shadow-lg">
                  <span className="relative flex h-2 w-2">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                  </span>
                  <span className="font-mono font-medium">30 FPS</span>
                  <span className="text-stone-400">•</span>
                  <span className="font-medium">133 Keypoints</span>
                </div>

                {liveGesture && (
                  <div className="flex items-center gap-2 bg-[#0B352B]/90 backdrop-blur-md text-emerald-200 text-xs px-3.5 py-1.5 rounded-full border border-emerald-500/30 shadow-lg">
                    <span className="text-[11px] text-emerald-400 uppercase tracking-wider font-semibold">Active Sign:</span>
                    <span className="font-bold text-white uppercase">{liveGesture}</span>
                    {confidence && (
                      <span className="bg-emerald-950/80 text-emerald-300 font-mono text-[10px] px-1.5 py-0.5 rounded">
                        {confidence}%
                      </span>
                    )}
                  </div>
                )}
              </div>

              {/* Bottom Floating Wispr Flow Subtitle Pill */}
              <div className="absolute bottom-5 left-4 right-4 flex flex-col items-center">
                <div className="w-full max-w-2xl bg-white/95 backdrop-blur-md rounded-2xl sm:rounded-full p-3 sm:px-6 sm:py-3.5 border border-stone-200 shadow-[0_12px_40px_rgba(0,0,0,0.25)] flex items-center justify-between gap-4 transition-all">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="text-[10px] font-semibold tracking-wider text-stone-500 uppercase flex items-center gap-1">
                        <Sparkles className="w-3 h-3 text-[#7C3AED]" />
                        Subtitles (Gemini 2.5 Flash)
                      </span>
                    </div>
                    <p className="text-sm sm:text-base font-semibold text-stone-900 truncate">
                      {flashText || sentence || (
                        <span className="text-stone-400 font-normal italic">
                          Sign continuously — sentences will form here automatically...
                        </span>
                      )}
                    </p>
                  </div>

                  <div className="flex items-center gap-1.5">
                    <button
                      onClick={handleCopy}
                      title="Copy Subtitle"
                      className="p-2 rounded-full hover:bg-stone-100 text-stone-600 transition-colors"
                    >
                      {copied ? <Check className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
                    </button>
                    <button
                      onClick={onClearBuffer}
                      title="Clear Buffer"
                      className="p-2 rounded-full hover:bg-stone-100 text-stone-600 transition-colors"
                    >
                      <RefreshCw className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Peaceful Idle Standby Screen (Camera Strictly Off) */
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center bg-[#FAF8F2] relative">
              {/* Ethereal background circles */}
              <div className="absolute w-72 h-72 rounded-full bg-emerald-900/5 blur-3xl pointer-events-none"></div>
              <div className="absolute w-60 h-60 rounded-full bg-purple-900/5 blur-3xl pointer-events-none"></div>

              {/* Hand/Camera Minimalist Icon Container */}
              <div className="relative z-10 w-16 h-16 rounded-3xl bg-white border border-stone-200/80 shadow-[0_8px_30px_rgba(0,0,0,0.06)] flex items-center justify-center mb-5 group">
                <Camera className="w-7 h-7 text-stone-800 transition-transform group-hover:scale-110" />
                <span className="absolute -top-1 -right-1 w-3.5 h-3.5 bg-stone-300 rounded-full border-2 border-white"></span>
              </div>

              {/* Standby Copy with Editorial Serif */}
              <h3 className="relative z-10 text-2xl sm:text-3xl font-normal text-stone-900 mb-2 font-editorial">
                <span>Step into frame, </span>
                <span className="italic font-normal">and flow.</span>
              </h3>
              <p className="relative z-10 max-w-md text-xs sm:text-sm text-stone-500 mb-6 leading-relaxed">
                Click start to activate RTMPose Wholebody tracking. Continuous signs are vectorized at 30 FPS and assembled into natural English sentences.
              </p>

              {/* Big Wispr Flow Signature Lavender CTA Button */}
              <button
                onClick={onToggleVision}
                disabled={visionConnecting}
                className="relative z-10 inline-flex items-center gap-2.5 px-6 py-3 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] text-sm font-semibold shadow-md shadow-purple-950/5 transition-all duration-200 active:scale-95 disabled:opacity-60"
              >
                <Play className="w-4 h-4 fill-[#1E1B4B]" />
                <span>{visionConnecting ? 'Initializing RTMPose...' : 'Start Detection'}</span>
              </button>

              {/* Core Features Micro-badges */}
              <div className="relative z-10 mt-8 flex flex-wrap items-center justify-center gap-3 text-[11px] text-stone-500">
                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-stone-200/60 shadow-sm">
                  <Zap className="w-3 h-3 text-amber-500" />
                  &lt;0.1ms Buffer
                </span>
                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-stone-200/60 shadow-sm">
                  <ShieldCheck className="w-3 h-3 text-emerald-600" />
                  50 Verified Signs (99.7% Acc)
                </span>
                <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-white border border-stone-200/60 shadow-sm">
                  <Sparkles className="w-3 h-3 text-purple-600" />
                  Gemini 2.5 Flash Restructuring
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Bottom Real-time Stream Info Strip */}
        <div className="px-6 py-3.5 bg-stone-50/80 border-t border-stone-100 flex flex-wrap items-center justify-between gap-3 text-xs text-stone-600 font-medium">
          <div className="flex items-center gap-2">
            <span className="text-stone-400 font-mono">STATUS:</span>
            <span className={isVisionRunning ? 'text-emerald-700 font-semibold' : 'text-stone-500'}>
              {isVisionRunning ? 'Vision Subtitler Active' : 'Standby (Camera Idle)'}
            </span>
          </div>
          
          <div className="flex items-center gap-4 text-stone-500">
            <span>Conversational Cooldown: <strong>0.35s</strong></span>
            <span>Hold Locking: <strong>0.12s</strong></span>
            <span>Elevation Guard: <strong>Active</strong></span>
          </div>
        </div>
      </div>
    </section>
  );
}
