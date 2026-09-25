import React, { useState, useEffect } from 'react';
import { Fingerprint, X, CheckCircle, Cpu, Zap, Database, Activity } from 'lucide-react';

export default function DiagnosticsBadge() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => setStatus(data))
      .catch((err) => console.error('Status fetch error:', err));
  }, []);

  return (
    <>
      {/* Floating Lilac Fingerprint Button (Wispr Flow Signature) */}
      <button
        onClick={() => setOpen(!open)}
        aria-label="System Diagnostics"
        className="fixed bottom-6 left-6 z-40 w-12 h-12 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] shadow-lg shadow-purple-950/10 flex items-center justify-center transition-all duration-200 hover:scale-105 active:scale-95"
      >
        <Fingerprint className="w-6 h-6 stroke-[1.8]" />
      </button>

      {/* Diagnostics Modal / Flyout */}
      {open && (
        <div className="fixed bottom-20 left-6 z-50 w-80 sm:w-96 bg-white/95 backdrop-blur-md rounded-3xl p-5 border border-stone-200/90 shadow-[0_20px_50px_rgba(0,0,0,0.15)] animate-in fade-in slide-in-from-bottom-3 duration-200">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-stone-100">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
              <h4 className="text-xs font-bold text-stone-900 uppercase tracking-wider">
                Lexis Architecture Status
              </h4>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="p-1 rounded-full hover:bg-stone-100 text-stone-400 hover:text-stone-700 transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between py-1 border-b border-stone-50">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-emerald-600" />
                Vision Tracker
              </span>
              <span className="font-medium text-stone-800">RTMPose (133 3D Points)</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-stone-50">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-amber-500" />
                Vectorized Buffer
              </span>
              <span className="font-mono text-emerald-700 font-bold">&lt;0.1ms Latency</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-stone-50">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-purple-600" />
                Continuous Model
              </span>
              <span className="font-medium text-stone-800">ExtraTrees (50 Words, 99.7%)</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-stone-50">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-blue-600" />
                Sign Studio
              </span>
              <span className="font-medium text-stone-800">Bi-GRU (2,414 Classes)</span>
            </div>

            <div className="flex items-center justify-between py-1 border-b border-stone-50">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-purple-600" />
                Sentence Restructuring
              </span>
              <span className="font-medium text-stone-800">Gemini 2.5 Flash</span>
            </div>

            <div className="flex items-center justify-between py-1">
              <span className="text-stone-500 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-emerald-600" />
                Database
              </span>
              <span className="font-medium text-stone-800">Neon Cloud PostgreSQL</span>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
