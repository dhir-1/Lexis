import React from 'react';
import { Check, Sparkles } from 'lucide-react';

export default function SurrealistRibbon({ liveSentence, lastGesture }) {
  // Use real detected text if available, otherwise display the iconic Wispr Flow floating thought
  const displayText = liveSentence 
    ? liveSentence 
    : "Where is the bathroom, please? I need water and help right now...";

  return (
    <div className="relative w-full max-w-5xl mx-auto pt-10 pb-8 px-4 text-center overflow-hidden">
      {/* Eyebrow Label */}
      <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-stone-200/50 text-[11px] font-semibold tracking-wider text-stone-600 uppercase mb-4">
        <Sparkles className="w-3 h-3 text-stone-700" />
        <span>Continuous ASL Vision & Voice Flow</span>
      </div>

      {/* Hero Editorial Serif Headline */}
      <h1 className="text-5xl sm:text-7xl font-normal tracking-tight text-stone-900 leading-[1.08] mb-5 font-editorial">
        <span>Don’t pause, </span>
        <span className="italic block sm:inline font-normal text-stone-800">just flow.</span>
      </h1>

      {/* Subtitle */}
      <p className="max-w-xl mx-auto text-base sm:text-lg text-stone-600 leading-relaxed font-normal">
        Real-time continuous ASL vision subtitling and multilingual speech translation powered by 133 Wholebody landmarks and Gemini 2.5 Flash.
      </p>

      {/* Wispr Flow Iconic Surrealist Tape / Ribbon */}
      <div className="relative mt-8 py-6 select-none pointer-events-none">
        {/* Curving Ribbon Track */}
        <div className="relative w-full flex items-center justify-center">
          <div className="relative z-10 inline-flex items-center gap-3 bg-[#111815] text-[#F3EFE6] px-6 py-2.5 rounded-full shadow-[0_12px_40px_rgba(0,0,0,0.18)] border border-stone-800 transform -rotate-1 hover:rotate-0 transition-transform">
            
            {/* Animated Waveform Pill */}
            <div className="flex items-center gap-[3px] bg-stone-900/90 px-2.5 py-1 rounded-full border border-stone-700/60">
              <span className="w-[3px] h-3 bg-white rounded-full animate-[wavePulse_0.8s_ease-in-out_infinite]"></span>
              <span className="w-[3px] h-5 bg-white rounded-full animate-[wavePulse_0.6s_ease-in-out_infinite_0.1s]"></span>
              <span className="w-[3px] h-2 bg-white rounded-full animate-[wavePulse_0.9s_ease-in-out_infinite_0.2s]"></span>
              <span className="w-[3px] h-4 bg-white rounded-full animate-[wavePulse_0.7s_ease-in-out_infinite_0.3s]"></span>
              <span className="w-[3px] h-3 bg-white rounded-full animate-[wavePulse_0.8s_ease-in-out_infinite_0.15s]"></span>
            </div>

            {/* Streaming Text */}
            <span className="text-xs sm:text-sm font-medium tracking-wide truncate max-w-[280px] sm:max-w-md text-stone-200">
              “{displayText}”
            </span>

            {/* Emerald Checkmark Badge */}
            <div className="flex items-center gap-1 bg-[#0B352B] text-emerald-300 text-[11px] font-semibold px-2.5 py-0.5 rounded-full border border-emerald-800/80 shadow-inner">
              <Check className="w-3 h-3 stroke-[3]" />
              <span>Gemini Polished</span>
            </div>
          </div>
        </div>

        {/* Subtle curved background graphic resembling the ribbon loop from Wispr Flow */}
        <svg
          className="absolute -top-6 left-1/2 -translate-x-1/2 w-full max-w-3xl h-28 pointer-events-none opacity-20 -z-0"
          viewBox="0 0 800 120"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M 20 60 C 200 120, 600 -20, 780 60"
            stroke="#171816"
            strokeWidth="1.5"
            strokeDasharray="4 4"
          />
        </svg>
      </div>
    </div>
  );
}
