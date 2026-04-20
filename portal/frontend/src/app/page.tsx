'use client';

import { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';

// Dynamic import for graph to avoid SSR issues with Cytoscape/D3
const GraphView = dynamic(() => import('@/components/GraphView'), { ssr: false });

interface Clip {
  clip: {
    clip_id: string;
    start_time_sec: number;
    end_time_sec: number;
  };
  vision?: {
    summary: string;
  };
}

interface ReasoningStep {
  step: string;
  content: string;
}

interface ReasoningResult {
  answer: string;
  reasoning_steps: ReasoningStep[];
  evidence_clips: string[];
}

export default function Home() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [query, setQuery] = useState('');
  const [reasoning, setReasoning] = useState<ReasoningResult | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('http://localhost:8000/api/clips')
      .then(res => res.json())
      .then(data => setClips(data));
  }, []);

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setReasoning(null);
    try {
      const res = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setReasoning(data);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <header className="glass-panel">
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div className="animate-pulse-slow" style={{ width: '12px', height: '12px', borderRadius: '50%', background: 'var(--accent-success)' }}></div>
          <h1 style={{ fontSize: '18px', fontWeight: 'bold', letterSpacing: '1px' }}>SYNPSE RAG <span style={{ color: 'var(--accent-primary)', fontSize: '12px' }}>V0.1-ALPHA</span></h1>
        </div>
        <div style={{ display: 'flex', gap: '24px', fontSize: '12px', color: 'var(--text-secondary)' }}>
          <div>VLM: <span style={{ color: 'var(--text-primary)' }}>QWEN2.5-VL</span></div>
          <div>ASR: <span style={{ color: 'var(--text-primary)' }}>COHERE</span></div>
          <div>GPU: <span style={{ color: 'var(--text-primary)' }}>M1 PRO (LOCAL)</span></div>
        </div>
      </header>

      <section className="sidebar glass-panel">
        <div style={{ padding: '20px', borderBottom: '1px solid var(--border-glass)' }}>
          <h2 style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '16px' }}>VIDEO TIMELINE</h2>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {clips.map((c, i) => (
              <div key={i} className="clip-card glass-panel">
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span className="badge" style={{ background: 'rgba(59, 130, 246, 0.2)', color: 'var(--accent-primary)' }}>{c.clip.clip_id}</span>
                  <span style={{ fontSize: '10px', color: 'var(--text-secondary)' }}>{c.clip.start_time_sec.toFixed(1)}s - {c.clip.end_time_sec.toFixed(1)}s</span>
                </div>
                <p style={{ fontSize: '12px', lineHeight: '1.4', color: 'var(--text-secondary)' }}>
                  {c.vision?.summary ? `${c.vision.summary.substring(0, 80)}...` : 'Processing or inference error...'}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="content-main">
        <div className="glass-panel glow-blue" style={{ flex: 1, padding: '20px', position: 'relative', overflow: 'hidden' }}>
          <h2 style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '16px' }}>KNOWLEDGE MAP (LAYERS 1 & 2)</h2>
          <div style={{ width: '100%', height: 'calc(100% - 40px)' }}>
            <GraphView />
          </div>
        </div>
      </section>

      <section className="query-panel">
        <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
          <div style={{ padding: '20px', borderBottom: '1px solid var(--border-glass)' }}>
            <h2 style={{ fontSize: '14px', color: 'var(--text-secondary)', marginBottom: '16px' }}>REASONING CONSOLE</h2>
            <form onSubmit={handleQuery} style={{ position: 'relative' }}>
              <input 
                type="text" 
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about the video content..."
                style={{ 
                  width: '100%', 
                  background: 'rgba(0,0,0,0.3)', 
                  border: '1px solid var(--border-glass)',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  color: 'white',
                  outline: 'none',
                  fontSize: '14px'
                }}
              />
              <button disabled={loading} style={{ 
                position: 'absolute', right: '8px', top: '8px', border: 'none', background: 'var(--accent-primary)', 
                color: 'white', padding: '4px 12px', borderRadius: '4px', cursor: 'pointer', fontSize: '12px'
              }}>
                {loading ? '...' : 'RUN'}
              </button>
            </form>
          </div>

          <div style={{ flex: 1, padding: '20px', overflowY: 'auto' }}>
            {reasoning ? (
              <>
                <div style={{ marginBottom: '24px' }}>
                  <div className="step-header">SUMMARY_ANSWER</div>
                  <p style={{ fontSize: '14px', lineHeight: '1.6' }}>{reasoning.answer}</p>
                </div>
                
                {reasoning.reasoning_steps.map((s: ReasoningStep, i: number) => (
                  <div key={i} className="reasoning-step">
                    <div className="step-header">[{s.step}]</div>
                    <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>{s.content}</div>
                  </div>
                ))}

                <div style={{ marginTop: '24px' }}>
                  <div className="step-header">GROUNDED_EVIDENCE</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px', marginTop: '8px' }}>
                    {reasoning.evidence_clips.map((cid: string) => (
                      <div key={cid} className="glass-panel" style={{ padding: '8px', fontSize: '11px', textAlign: 'center' }}>
                        Clip {cid}
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)', fontSize: '12px' }}>
                Waiting for input...
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
}
