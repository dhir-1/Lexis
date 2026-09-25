import React, { useState, useEffect } from 'react';
import { Search, ShieldCheck, Sparkles, BookOpen, ExternalLink } from 'lucide-react';

const VERIFIED_50 = [
  "again", "baby", "bathroom", "book", "brother", "car", "drink", "drive",
  "eat", "family", "father", "fine", "friend", "goodbye", "happy", "hello",
  "help", "home", "house", "how", "i love you", "like", "man", "more",
  "mother", "my", "name", "no", "play", "please", "sad", "school",
  "sister", "sorry", "stop", "student", "teacher", "thank you", "time", "want",
  "water", "what", "when", "where", "who", "why", "woman", "work",
  "yes", "you"
];

export default function LexiconSection() {
  const [query, setQuery] = useState('');
  const [filterMode, setFilterMode] = useState('verified'); // 'verified' or 'all'
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (filterMode === 'all' && query.trim().length > 0) {
      const timer = setTimeout(async () => {
        setLoading(true);
        try {
          const res = await fetch(`/api/studio/search?q=${encodeURIComponent(query)}&limit=36`);
          const data = await res.json();
          setSearchResults(data.results || []);
        } catch (err) {
          console.error('Search error:', err);
        } finally {
          setLoading(false);
        }
      }, 200);
      return () => clearTimeout(timer);
    }
  }, [query, filterMode]);

  const displayList = filterMode === 'verified'
    ? VERIFIED_50.filter((w) => w.toLowerCase().includes(query.toLowerCase()))
    : searchResults.map((r) => r.sign);

  return (
    <section className="w-full max-w-5xl mx-auto px-4 py-8">
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-stone-200/50 text-[11px] font-semibold tracking-wider text-stone-600 uppercase mb-3">
          <BookOpen className="w-3 h-3 text-stone-700" />
          <span>ASL Vocabulary Matrix</span>
        </div>
        <h2 className="text-4xl sm:text-5xl font-normal tracking-tight text-stone-900 leading-tight font-editorial">
          <span>2,414 signs, </span>
          <span className="italic font-normal">at your fingertips.</span>
        </h2>
        <p className="max-w-md mx-auto text-sm sm:text-base text-stone-500 mt-2 font-normal">
          Explore the 50 verified conversational signs with 99.7% cross-validation accuracy, plus 2,414 deep-learning WLASL vocabulary classes.
        </p>
      </div>

      {/* Search and Segmented Filter Bar */}
      <div className="bg-white rounded-3xl p-4 border border-stone-200/80 shadow-[0_10px_30px_rgba(0,0,0,0.03)] flex flex-col sm:flex-row items-center justify-between gap-4 mb-6">
        {/* Search Input */}
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-stone-400 absolute left-3.5 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search sign (e.g. bathroom, help)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-9 pr-4 py-2 text-xs sm:text-sm bg-stone-50 border border-stone-200 rounded-full focus:outline-none focus:border-stone-400 text-stone-800 placeholder-stone-400"
          />
        </div>

        {/* Filter Switcher */}
        <div className="flex items-center bg-[#F4F1EA] p-1 rounded-full border border-stone-200 text-xs font-medium w-full sm:w-auto">
          <button
            onClick={() => setFilterMode('verified')}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-full transition-all ${
              filterMode === 'verified'
                ? 'bg-white text-stone-900 font-semibold shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-600" />
            <span>Verified Core (50)</span>
          </button>

          <button
            onClick={() => {
              setFilterMode('all');
              if (!query) setQuery('a');
            }}
            className={`flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-3.5 py-1.5 rounded-full transition-all ${
              filterMode === 'all'
                ? 'bg-white text-stone-900 font-semibold shadow-sm'
                : 'text-stone-600 hover:text-stone-900'
            }`}
          >
            <Sparkles className="w-3.5 h-3.5 text-purple-600" />
            <span>All WLASL (2,414)</span>
          </button>
        </div>
      </div>

      {/* Grid of Signs */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
        {displayList.map((sign, i) => (
          <div
            key={i}
            className="group bg-white hover:bg-[#FAF8F2] p-4 rounded-2xl border border-stone-200/70 hover:border-stone-400/80 shadow-sm transition-all duration-200 flex flex-col justify-between"
          >
            <div className="flex items-start justify-between">
              <span className="text-xs font-mono text-stone-400">#{i + 1}</span>
              {filterMode === 'verified' && (
                <span className="w-2 h-2 rounded-full bg-emerald-500" title="99.7% Verified Accuracy"></span>
              )}
            </div>

            <div className="my-2">
              <h4 className="text-base font-semibold text-stone-900 capitalize tracking-tight group-hover:text-stone-950">
                {sign}
              </h4>
              <p className="text-[11px] text-stone-400">
                {filterMode === 'verified' ? 'Core conversational' : 'WLASL Studio Class'}
              </p>
            </div>

            <div className="pt-2 border-t border-stone-100 flex items-center justify-between text-[11px] text-stone-500">
              <span>{filterMode === 'verified' ? '99.7% Acc' : 'Bi-GRU'}</span>
              <span className="text-stone-400 group-hover:text-stone-800 transition-colors">
                ASL
              </span>
            </div>
          </div>
        ))}
      </div>

      {displayList.length === 0 && (
        <div className="text-center py-12 text-stone-400 text-sm">
          No matching signs found. Try another search query.
        </div>
      )}
    </section>
  );
}
