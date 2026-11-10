// SVG chart components fed from window.DATA
const C = window.DATA.colors;
const col = (n) => C[n] || "#9ba6c9";
const HL = new Set(["CreDRO-Soft"]);

// Horizontal bar chart.  rows: [{name, v}], lowerBetter marks min as best
function HBar({rows, max, fmt = (v) => v.toFixed(3), lowerBetter = false, width = 760, barH = 30, labelW = 170, title}) {
  const H = rows.length * (barH + 8) + 40;
  const best = lowerBetter ? Math.min(...rows.map(r => r.v)) : Math.max(...rows.map(r => r.v));
  const W = width - labelW - 90;
  const mx = max ?? Math.max(...rows.map(r => r.v)) * 1.08;
  return (
    <svg viewBox={`0 0 ${width} ${H}`} style={{width: "100%", height: "auto"}}>
      {title && <text x={labelW} y={18} fill="#9ba6c9" fontSize="14" className="mono">{title}</text>}
      {rows.map((r, i) => {
        const y = 32 + i * (barH + 8); const w = Math.max(2, (r.v / mx) * W); const hl = HL.has(r.name);
        return (
          <g key={r.name} opacity={1}>
            <text x={labelW - 12} y={y + barH / 2 + 5} textAnchor="end" fill={hl ? "#8b97ff" : "#e9edf9"} fontSize="16" fontWeight={hl ? 800 : 500}>{r.name}</text>
            <rect x={labelW} y={y} width={W} height={barH} rx="7" fill="rgba(255,255,255,.04)" />
            <rect x={labelW} y={y} width={w} height={barH} rx="7" fill={col(r.name)} opacity={hl ? 1 : .75}>
              <animate attributeName="width" from="0" to={w} dur="0.7s" fill="freeze" />
            </rect>
            <text x={labelW + w + 10} y={y + barH / 2 + 5} fill={r.v === best ? "#3dd6c6" : "#9ba6c9"} fontSize="15" fontWeight={r.v === best ? 800 : 500} className="mono">{fmt(r.v)}{r.v === best ? " ★" : ""}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Dot + error bar chart with per-fold points (LOTO)
function DotErr({names, width = 780}) {
  const d = window.DATA.loto; const rowH = 44, labelW = 160, left = labelW, W = width - labelW - 40;
  const lo = 0.70, hi = 0.96; const x = (v) => left + ((v - lo) / (hi - lo)) * W;
  const H = names.length * rowH + 60;
  return (
    <svg viewBox={`0 0 ${width} ${H}`} style={{width: "100%", height: "auto"}}>
      {[0.7, 0.75, 0.8, 0.85, 0.9, 0.95].map(t => (
        <g key={t}><line x1={x(t)} x2={x(t)} y1={28} y2={H - 30} stroke="rgba(255,255,255,.08)" />
          <text x={x(t)} y={H - 12} fill="#6b7598" fontSize="13" textAnchor="middle" className="mono">{t.toFixed(2)}</text></g>
      ))}
      <text x={x(0.825)} y={14} fill="#9ba6c9" fontSize="13" textAnchor="middle" className="mono">AUROC unseen target (upper entropy), 5 LOTO folds</text>
      {names.map((n, i) => {
        const r = d[n]; const y = 40 + i * rowH; const hl = HL.has(n);
        return (
          <g key={n}>
            <text x={labelW - 14} y={y + 5} textAnchor="end" fill={hl ? "#8b97ff" : "#e9edf9"} fontSize="16" fontWeight={hl ? 800 : 500}>{n}</text>
            <line x1={x(r.m - r.s)} x2={x(r.m + r.s)} y1={y} y2={y} stroke={col(n)} strokeWidth="3" strokeLinecap="round" opacity=".55" />
            {r.f && r.f.map((v, k) => <circle key={k} cx={x(v)} cy={y} r="4.5" fill="none" stroke={col(n)} strokeWidth="1.6" opacity=".9" />)}
            <circle cx={x(r.m)} cy={y} r="8" fill={col(n)} className={hl ? "glow" : ""} />
            <text x={x(r.m)} y={y - 13} fill={hl ? "#8b97ff" : "#9ba6c9"} fontSize="13" textAnchor="middle" className="mono" fontWeight={hl ? 800 : 500}>{r.m.toFixed(3)} ± {r.s.toFixed(3)}</text>
          </g>
        );
      })}
    </svg>
  );
}

// Line chart for ensemble-size ablation
function Lines({metric, yl, yh, title, width = 720, height = 360, fmt = v => v.toFixed(2)}) {
  const a = window.DATA.abl; const names = ["CreWrap-DE", "CreDRO", "CreDRO-Soft", "CreWrap-Soft"];
  const Ms = a.M.filter(m => m <= 8); const L = 60, R = 120, T = 34, B = 44;
  const x = (m) => L + ((m - 1) / 7) * (width - L - R); const y = (v) => T + (1 - (v - yl) / (yh - yl)) * (height - T - B);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} style={{width: "100%", height: "auto"}}>
      <text x={L} y={18} fill="#9ba6c9" fontSize="14" className="mono">{title}</text>
      {[0, .25, .5, .75, 1].map(f => { const v = yl + f * (yh - yl); return (
        <g key={f}><line x1={L} x2={width - R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,.08)" />
          <text x={L - 8} y={y(v) + 4} fill="#6b7598" fontSize="12" textAnchor="end" className="mono">{fmt(v)}</text></g>); })}
      {Ms.map(m => <text key={m} x={x(m)} y={height - 14} fill="#6b7598" fontSize="13" textAnchor="middle" className="mono">M={m}</text>)}
      {(() => {
        const series = names.map(n => { const pts = Ms.map((m, i) => [m, a[n][metric][i]]).filter(p => p[1] != null); return {n, pts, ly: y(pts[pts.length - 1][1])}; });
        const sorted = [...series].sort((p, q) => p.ly - q.ly); // de-overlap end labels (min 16px apart)
        for (let i = 1; i < sorted.length; i++) if (sorted[i].ly - sorted[i - 1].ly < 16) sorted[i].ly = sorted[i - 1].ly + 16;
        return series.map(({n, pts, ly}) => { const path = pts.map((p, i) => `${i ? "L" : "M"}${x(p[0])},${y(p[1])}`).join(" "); const hl = HL.has(n); const lx = x(pts[pts.length - 1][0]);
          return (<g key={n}>
            <path d={path} fill="none" stroke={col(n)} strokeWidth={hl ? 4 : 2.5} strokeLinejoin="round" opacity={hl ? 1 : .8} />
            {pts.map(p => <circle key={p[0]} cx={x(p[0])} cy={y(p[1])} r={hl ? 6 : 4.5} fill={col(n)} />)}
            <text x={lx + 10} y={ly + 4} fill={col(n)} fontSize="13" fontWeight="700">{n}</text>
          </g>); });
      })()}
    </svg>
  );
}

// Grouped bars: two values per method
function Grouped({rows, labels, colors = ["#94a3b8", "#8b97ff"], max, width = 760, fmt = v => v.toFixed(3), title}) {
  const barH = 14, gap = 22, labelW = 160; const W = width - labelW - 90; const mx = max ?? Math.max(...rows.flatMap(r => r.v.filter(x => x != null))) * 1.1;
  const H = rows.length * (barH * 2 + gap) + 70;
  return (
    <svg viewBox={`0 0 ${width} ${H}`} style={{width: "100%", height: "auto"}}>
      <text x={labelW} y={16} fill="#9ba6c9" fontSize="14" className="mono">{title}</text>
      {labels.map((l, k) => <g key={l}><rect x={labelW + k * 270} y={28} width="12" height="12" rx="3" fill={colors[k]} opacity={k ? 1 : .55} /><text x={labelW + 18 + k * 270} y={39} fill="#9ba6c9" fontSize="13">{l}</text></g>)}
      {rows.map((r, i) => { const y0 = 56 + i * (barH * 2 + gap); const hl = HL.has(r.name); return (
        <g key={r.name}>
          <text x={labelW - 12} y={y0 + barH + 4} textAnchor="end" fill={hl ? "#8b97ff" : "#e9edf9"} fontSize="15" fontWeight={hl ? 800 : 500}>{r.name}</text>
          {r.v.map((v, k) => v == null ? <text key={k} x={labelW + 4} y={y0 + k * (barH + 2) + barH - 2} fill="#6b7598" fontSize="12" className="mono">—</text> : (
            <g key={k}><rect x={labelW} y={y0 + k * (barH + 2)} width={(v / mx) * W} height={barH} rx="4" fill={colors[k]} opacity={k ? 1 : .55} />
              <text x={labelW + (v / mx) * W + 8} y={y0 + k * (barH + 2) + barH - 2} fill="#9ba6c9" fontSize="12.5" className="mono">{fmt(v)}</text></g>))}
        </g>); })}
    </svg>
  );
}
