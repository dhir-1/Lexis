import React from 'react';
import { Camera, Square, Mic, Video, BookOpen, Clock, Activity } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, isVisionRunning, onToggleVision, visionConnecting }) {
  const tabs = [
    { id: 'vision', label: 'Continuous Vision', icon: Camera },
    { id: 'voice', label: 'Voice Flow', icon: Mic },
    { id: 'studio', label: 'Sign Studio', icon: Video },
    { id: 'lexicon', label: '2,414 Lexicon', icon: BookOpen },
    { id: 'history', label: 'History', icon: Clock },
  ];

  return (
    <header className="sticky top-4 z-50 w-full px-4 sm:px-6 max-w-6xl mx-auto">
      <nav className="flex items-center justify-between bg-white/90 backdrop-blur-md border border-stone-200/90 rounded-full px-4 py-2.5 shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all">
        {/* Left: Brand Identity */}
        <div 
          onClick={() => setActiveTab('vision')} 
          className="flex items-center gap-2.5 cursor-pointer pl-1 group"
        >
          <div className="flex items-end gap-[3px] h-5 py-0.5">
            <span className="w-1 bg-stone-900 h-4 rounded-full group-hover:scale-y-110 transition-transform"></span>
            <span className="w-1 bg-stone-900 h-2.5 rounded-full group-hover:scale-y-90 transition-transform"></span>
            <span className="w-1 bg-stone-900 h-5 rounded-full group-hover:scale-y-125 transition-transform"></span>
            <span className="w-1 bg-stone-900 h-3.5 rounded-full group-hover:scale-y-105 transition-transform"></span>
          </div>
          <span className="font-semibold text-lg tracking-tight text-stone-900">Lexis</span>
          <span className="hidden sm:inline-block text-[11px] font-medium px-2 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200/60">
            Flow
          </span>
        </div>

        {/* Center: Segmented Mode Selector */}
        <div className="hidden md:flex items-center bg-[#F4F1EA] p-1 rounded-full border border-stone-200/60 text-xs font-medium">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition-all duration-200 ${
                  isActive
                    ? 'bg-white text-stone-900 shadow-sm shadow-stone-300/50 font-semibold'
                    : 'text-stone-600 hover:text-stone-900 hover:bg-stone-200/50'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>

        {/* Right: Wispr Flow signature Lavender CTA Button */}
        <div className="flex items-center gap-2">
          {isVisionRunning ? (
            <button
              onClick={onToggleVision}
              disabled={visionConnecting}
              className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-rose-50 text-rose-700 hover:bg-rose-100 border border-rose-200/80 text-xs font-semibold shadow-sm transition-all duration-150 active:scale-95 disabled:opacity-50"
            >
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-rose-600"></span>
              </span>
              <span>Stop Detection</span>
            </button>
          ) : (
            <button
              onClick={onToggleVision}
              disabled={visionConnecting}
              className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#EADDFE] hover:bg-[#E1CFFC] text-[#1E1B4B] border border-[#D8C2F8] text-xs font-semibold shadow-sm transition-all duration-150 active:scale-95 disabled:opacity-50"
            >
              <Activity className="w-3.5 h-3.5 text-[#4338CA]" />
              <span>{visionConnecting ? 'Starting...' : 'Start Detection'}</span>
            </button>
          )}
        </div>
      </nav>

      {/* Mobile Sub-Navigation Bar */}
      <div className="flex md:hidden items-center justify-around bg-white/90 backdrop-blur-md mt-2 p-1 rounded-full border border-stone-200 shadow-sm text-xs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`p-2 rounded-full transition-colors ${
                isActive ? 'bg-[#FAF8F2] text-stone-900 font-bold' : 'text-stone-500'
              }`}
              title={tab.label}
            >
              <Icon className="w-4 h-4" />
            </button>
          );
        })}
      </div>
    </header>
  );
}
