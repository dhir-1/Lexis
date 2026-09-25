import React, { useState, useEffect, useRef } from 'react';
import Navbar from './components/Navbar';
import SurrealistRibbon from './components/SurrealistRibbon';
import VisionSection from './components/VisionSection';
import VoiceSection from './components/VoiceSection';
import SignStudioSection from './components/SignStudioSection';
import LexiconSection from './components/LexiconSection';
import HistorySection from './components/HistorySection';
import DiagnosticsBadge from './components/DiagnosticsBadge';

export default function App() {
  const [activeTab, setActiveTab] = useState('vision');
  const [isVisionRunning, setIsVisionRunning] = useState(false);
  const [visionConnecting, setVisionConnecting] = useState(false);
  const [currentEvent, setCurrentEvent] = useState(null);

  const wsRef = useRef(null);

  // Check initial server vision status on mount without auto-starting
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => {
        if (data.vision_stream_running) {
          setIsVisionRunning(true);
          initWebSocket();
        }
      })
      .catch((err) => console.log('Status probe:', err));

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const initWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host || '127.0.0.1:8000';
    const ws = new WebSocket(`${protocol}//${host}/ws/vision`);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        setCurrentEvent(payload);
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };

    ws.onerror = (err) => {
      console.warn('WebSocket warning:', err);
    };

    ws.onclose = () => {
      console.log('WebSocket closed');
    };
  };

  const toggleVision = async () => {
    setVisionConnecting(true);
    try {
      if (!isVisionRunning) {
        // Start Vision
        const res = await fetch('/api/vision/start', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          setIsVisionRunning(true);
          initWebSocket();
        }
      } else {
        // Stop Vision
        await fetch('/api/vision/stop', { method: 'POST' });
        if (wsRef.current) {
          wsRef.current.close();
          wsRef.current = null;
        }
        setIsVisionRunning(false);
        setCurrentEvent(null);
      }
    } catch (err) {
      console.error('Toggle vision error:', err);
    } finally {
      setVisionConnecting(false);
    }
  };

  const handleClearBuffer = () => {
    setCurrentEvent((prev) => (prev ? { ...prev, sentence: '', flash_text: '' } : null));
  };

  return (
    <div className="min-h-screen bg-[#FAF8F2] text-[#171816] flex flex-col justify-between selection:bg-[#EADDFE] selection:text-[#1E1B4B]">
      {/* Top Floating Pill Navbar */}
      <div className="pt-4">
        <Navbar
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          isVisionRunning={isVisionRunning}
          onToggleVision={toggleVision}
          visionConnecting={visionConnecting}
        />
      </div>

      {/* Main Content Area */}
      <main className="flex-1 pb-16">
        {/* Surrealist Ribbon Hero Banner */}
        <SurrealistRibbon
          liveSentence={currentEvent?.flash_text || currentEvent?.sentence}
          lastGesture={currentEvent?.current_gesture}
        />

        {/* Tab 1: Continuous Vision Subtitler */}
        {activeTab === 'vision' && (
          <VisionSection
            isVisionRunning={isVisionRunning}
            onToggleVision={toggleVision}
            visionConnecting={visionConnecting}
            currentEvent={currentEvent}
            onClearBuffer={handleClearBuffer}
          />
        )}

        {/* Tab 2: Multilingual Voice Flow */}
        {activeTab === 'voice' && <VoiceSection />}

        {/* Tab 3: Sign Studio & Shazam for ASL */}
        {activeTab === 'studio' && <SignStudioSection />}

        {/* Tab 4: 2,414 Lexicon */}
        {activeTab === 'lexicon' && <LexiconSection />}

        {/* Tab 5: History */}
        {activeTab === 'history' && <HistorySection />}
      </main>

      {/* Bottom Floating Lilac Fingerprint Diagnostics Button */}
      <DiagnosticsBadge />

      {/* Aesthetic Footer */}
      <footer className="w-full max-w-5xl mx-auto px-6 py-8 border-t border-stone-200/60 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-stone-500">
        <div className="flex items-center gap-2">
          <span className="font-semibold text-stone-800">Lexis</span>
          <span>•</span>
          <span>Continuous ASL Vision & Voice Flow</span>
        </div>
        <div className="flex items-center gap-4 text-stone-400">
          <span>RTMPose Wholebody</span>
          <span>•</span>
          <span>Whisper Large V3</span>
          <span>•</span>
          <span>Gemini 2.5 Flash</span>
        </div>
      </footer>
    </div>
  );
}
