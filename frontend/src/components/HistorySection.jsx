import React, { useState, useEffect } from 'react';
import { Database, RefreshCw, Clock, ArrowUpRight, Camera, Mic, Film } from 'lucide-react';

export default function HistorySection() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/history?limit=30');
      const data = await res.json();
      setHistory(data.history || []);
    } catch (err) {
      console.error('Failed to load history:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <section className="w-full max-w-5xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-stone-200/50 text-[11px] font-semibold tracking-wider text-stone-600 uppercase mb-3">
          <Database className="w-3 h-3 text-stone-700" />
          <span>Neon Cloud PostgreSQL</span>
        </div>
        <h2 className="text-4xl sm:text-5xl font-normal tracking-tight text-stone-900 leading-tight font-editorial">
          <span>Translation </span>
          <span className="italic font-normal">telemetry.</span>
        </h2>
        <p className="max-w-md mx-auto text-sm sm:text-base text-stone-500 mt-2 font-normal">
          Immutable logs persisted to Neon Cloud PostgreSQL for continuous ASL sentences and multilingual voice dictation.
        </p>
      </div>

      <div className="bg-white rounded-3xl p-6 border border-stone-200/80 shadow-[0_10px_30px_rgba(0,0,0,0.03)]">
        <div className="flex items-center justify-between pb-4 mb-4 border-b border-stone-100">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-xs font-semibold text-stone-800">Live PostgreSQL Logs ({history.length})</span>
          </div>

          <button
            onClick={fetchHistory}
            disabled={loading}
            className="flex items-center gap-1.5 text-xs text-stone-600 hover:text-stone-900 font-medium px-3 py-1.5 rounded-full hover:bg-stone-100 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        {history.length === 0 ? (
          <div className="text-center py-12 text-stone-400 text-sm">
            {loading ? 'Fetching logs from Neon DB...' : 'No translation logs recorded yet. Start detection or speak to populate logs.'}
          </div>
        ) : (
          <div className="divide-y divide-stone-100">
            {history.map((item) => (
              <div key={item.id} className="py-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:bg-stone-50/60 px-3 rounded-2xl transition-colors">
                <div className="flex items-start gap-3">
                  <div className={`p-2 rounded-xl mt-0.5 ${
                    item.input_type === 'vision' ? 'bg-purple-50 text-purple-700' : 'bg-emerald-50 text-emerald-700'
                  }`}>
                    {item.input_type === 'vision' ? <Camera className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
                  </div>

                  <div>
                    <p className="text-sm font-semibold text-stone-900 leading-snug">
                      {item.translated_text}
                    </p>
                    {item.raw_text && (
                      <p className="text-xs text-stone-400 mt-0.5 font-mono">
                        Raw: {item.raw_text}
                      </p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs text-stone-400 font-mono self-end sm:self-center">
                  {item.detected_language && (
                    <span className="px-2 py-0.5 rounded-full bg-stone-100 text-stone-600 text-[10px] uppercase font-semibold">
                      {item.detected_language}
                    </span>
                  )}
                  {item.confidence_score && (
                    <span className="text-stone-500">
                      {Math.round(item.confidence_score * 100)}%
                    </span>
                  )}
                  <span>
                    {new Date(item.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
