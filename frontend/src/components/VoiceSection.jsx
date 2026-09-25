import React, { useState, useRef } from 'react';
import { Mic, Square, Upload, Sparkles, Volume2, Globe, Clock, Check, Copy } from 'lucide-react';

export default function VoiceSection() {
  const [isRecording, setIsRecording] = useState(false);
  const [loading, setLoading] = useState(false);
  const [audioResult, setAudioResult] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // Start recording using browser MediaRecorder
  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        // Stop all tracks to release mic
        stream.getTracks().forEach((track) => track.stop());
        await sendAudioToBackend(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('Microphone error:', err);
      setError('Could not access microphone. Please grant browser permissions.');
    }
  };

  // Stop recording and trigger translation
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Upload custom audio file
  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (file) {
      await sendAudioToBackend(file);
    }
  };

  const sendAudioToBackend = async (audioBlobOrFile) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('file', audioBlobOrFile, 'recording.webm');

      const response = await fetch('/api/audio/translate', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok && data.success) {
        setAudioResult(data);
      } else {
        setError(data.detail || 'Translation failed.');
      }
    } catch (err) {
      console.error('Translation network error:', err);
      setError('Connection error with Lexis backend.');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (audioResult?.english_translation) {
      navigator.clipboard.writeText(audioResult.english_translation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section className="w-full max-w-4xl mx-auto px-4 py-8">
      {/* Editorial Headline */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-stone-200/50 text-[11px] font-semibold tracking-wider text-stone-600 uppercase mb-3">
          <Globe className="w-3 h-3 text-stone-700" />
          <span>Multilingual Speech Flow</span>
        </div>
        <h2 className="text-4xl sm:text-5xl font-normal tracking-tight text-stone-900 leading-tight font-editorial">
          <span>Don’t type, </span>
          <span className="italic font-normal">just speak.</span>
        </h2>
        <p className="max-w-md mx-auto text-sm sm:text-base text-stone-500 mt-2 font-normal">
          Speak in Hindi, Spanish, French, or 99+ languages. Native Whisper Large V3 translates directly into English in ~630ms.
        </p>
      </div>

      {/* Main Surrealist Voice Interaction Card */}
      <div className="bg-white rounded-[36px] border border-stone-200/80 shadow-[0_20px_60px_-15px_rgba(0,0,0,0.05)] p-8 sm:p-10 flex flex-col items-center relative overflow-hidden">
        
        {/* Animated Sound Wave or Idle Mic Orb */}
        <div className="relative mb-8 mt-2">
          {/* Concentric ripples while recording */}
          {isRecording && (
            <>
              <div className="absolute inset-0 -m-4 rounded-full bg-rose-500/10 animate-ping"></div>
              <div className="absolute inset-0 -m-8 rounded-full bg-rose-500/5 animate-pulse"></div>
            </>
          )}

          <button
            onClick={isRecording ? stopRecording : startRecording}
            disabled={loading}
            className={`relative z-10 w-24 h-24 rounded-full flex flex-col items-center justify-center transition-all duration-300 shadow-xl ${
              isRecording
                ? 'bg-rose-600 text-white scale-105 shadow-rose-600/30'
                : 'bg-[#171816] text-[#FAF8F2] hover:scale-105 shadow-stone-900/20 active:scale-95'
            }`}
          >
            {isRecording ? (
              <Square className="w-8 h-8 fill-white" />
            ) : (
              <Mic className="w-8 h-8 stroke-[2.2]" />
            )}
          </button>
        </div>

        {/* State label & CTA */}
        <div className="text-center mb-6">
          <p className="text-base font-semibold text-stone-900">
            {isRecording
              ? 'Listening... Speak in Hindi or any language'
              : loading
              ? 'Translating with Whisper Large V3...'
              : 'Tap to speak or upload audio'}
          </p>
          <p className="text-xs text-stone-400 mt-1">
            {isRecording ? 'Click button to stop and translate' : 'Native zero-shot English translation'}
          </p>
        </div>

        {/* Floating Waveform Bars when recording */}
        {isRecording && (
          <div className="flex items-center gap-1.5 h-8 mb-6">
            {[40, 70, 95, 30, 85, 100, 45, 90, 60, 100, 50, 80, 65, 35].map((height, i) => (
              <span
                key={i}
                style={{ height: `${height}%` }}
                className="w-1 bg-[#171816] rounded-full animate-[wavePulse_0.6s_ease-in-out_infinite]"
              ></span>
            ))}
          </div>
        )}

        {/* File Upload Option */}
        <div className="flex items-center gap-2 mb-4">
          <label className="cursor-pointer text-xs font-medium text-stone-500 hover:text-stone-800 flex items-center gap-1.5 px-3 py-1.5 rounded-full border border-stone-200 hover:border-stone-300 transition-colors">
            <Upload className="w-3.5 h-3.5" />
            <span>Upload Audio File (.wav, .mp3, .webm)</span>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>
        </div>

        {/* Error notification */}
        {error && (
          <div className="w-full max-w-md p-3 rounded-2xl bg-rose-50 border border-rose-200 text-rose-700 text-xs text-center mb-4">
            {error}
          </div>
        )}

        {/* Translation Results Card */}
        {audioResult && (
          <div className="w-full mt-6 bg-[#FCFBF7] rounded-3xl p-6 border border-stone-200/90 shadow-sm animate-in fade-in duration-300">
            <div className="flex items-center justify-between border-b border-stone-200/60 pb-3 mb-4">
              <div className="flex items-center gap-2">
                <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-[#0B352B] text-emerald-300 uppercase tracking-wide">
                  {audioResult.detected_language || 'Detected'}
                </span>
                <span className="text-xs text-stone-500 flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {audioResult.direct_latency_ms}ms direct latency
                </span>
              </div>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-xs text-stone-600 hover:text-stone-900 font-medium px-2 py-1 rounded-lg hover:bg-stone-100 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>

            {/* Original Spoken Text */}
            <div className="mb-4">
              <span className="text-[11px] font-semibold text-stone-400 uppercase tracking-wider block mb-1">
                Original Speech:
              </span>
              <p className="text-sm text-stone-700 font-medium italic">
                “{audioResult.original_speech}”
              </p>
            </div>

            {/* Polished English Translation */}
            <div>
              <span className="text-[11px] font-semibold text-purple-700 uppercase tracking-wider block mb-1 flex items-center gap-1">
                <Sparkles className="w-3 h-3 text-purple-600" />
                English Translation:
              </span>
              <p className="text-lg font-semibold text-stone-900 leading-snug">
                {audioResult.english_translation}
              </p>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
