'use client';

import { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';

export default function GraphView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const fetchGraph = async () => {
      const res = await fetch('http://localhost:8000/api/graph');
      const elements = await res.json();

      cyRef.current = cytoscape({
        container: containerRef.current,
        elements: elements,
        style: [
          {
            selector: 'node',
            style: {
              'background-color': '#3b82f6',
              'label': 'data(label)',
              'color': '#fff',
              'font-size': '10px',
              'text-valign': 'center',
              'text-halign': 'center',
              'width': 40,
              'height': 40,
              'border-width': 2,
              'border-color': 'rgba(255,255,255,0.2)'
            }
          },
          {
            selector: '.layer1',
            style: {
              'background-color': '#10b981',
              'shape': 'round-rectangle',
              'width': 60,
              'height': 30
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 1,
              'line-color': 'rgba(255,255,255,0.2)',
              'target-arrow-color': 'rgba(255,255,255,0.2)',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'label': 'data(label)',
              'font-size': '8px',
              'color': 'rgba(255,255,255,0.5)',
              'text-rotation': 'autorotate'
            }
          },
          {
            selector: '.semantic_edge',
            style: {
              'line-color': '#a855f7',
              'width': 2,
              'line-style': 'dashed'
            }
          }
        ],
        layout: {
          name: 'grid',
          rows: 3
        }
      });
    };

    fetchGraph();

    return () => {
      if (cyRef.current) {
        cyRef.current.destroy();
      }
    };
  }, []);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
}
