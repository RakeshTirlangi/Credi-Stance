// Hand-built SVG diagrams for the deck.
const CLS = {F: "#4ade80", A: "#fb7185", N: "#94a3b8"};
const INK = "#e9edf9", MUTED = "#9ba6c9", DIM = "#6b7598", ACC = "#8b97ff", ACC2 = "#3dd6c6", GOLD = "#ffc46b", ROSE = "#ff6f9c", VIO = "#c084fc";

// ---------- geometry helpers for the 3-class simplex ----------
function simplexVerts(cx, cy, r) { // FAVOR top, AGAINST bottom-left, NONE bottom-right
  return [[cx, cy - r], [cx - r * 0.866, cy + r * 0.5], [cx + r * 0.866, cy + r * 0.5]];
}
function bary(p, V) { return [p[0] * V[0][0] + p[1] * V[1][0] + p[2] * V[2][0], p[0] * V[0][1] + p[1] * V[1][1] + p[2] * V[2][1]]; }
function credalPolygon(l, u) { // vertices of {p: l<=p<=u, sum p = 1}
  const pts = [];
  for (let i = 0; i < 3; i++) for (let j = i + 1; j < 3; j++) {
    const k = 3 - i - j;
    for (const a of [l[i], u[i]]) for (const b of [l[j], u[j]]) {
      const c = 1 - a - b; if (c >= l[k] - 1e-9 && c <= u[k] + 1e-9) { const p = [0, 0, 0]; p[i] = a; p[j] = b; p[k] = c; pts.push(p); }
    }
  }
  const uniq = []; pts.forEach(p => { if (!uniq.some(q => Math.abs(q[0] - p[0]) + Math.abs(q[1] - p[1]) + Math.abs(q[2] - p[2]) < 1e-6)) uniq.push(p); });
  const c = uniq.reduce((s, p) => [s[0] + p[0] / uniq.length, s[1] + p[1] / uniq.length, s[2] + p[2] / uniq.length], [0, 0, 0]);
  return uniq.sort((p, q) => Math.atan2(p[1] - c[1], p[0] - c[0]) - Math.atan2(q[1] - c[1], q[0] - c[0]));
}

function Simplex({cx, cy, r, l, u, points = [], point, label, fill = ACC, showLabels = true, extra}) {
  const V = simplexVerts(cx, cy, r);
  const poly = l ? credalPolygon(l, u).map(p => bary(p, V)) : null;
  return (
    <g>
      <polygon points={V.map(v => v.join(",")).join(" ")} fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.25)" strokeWidth="1.5" />
      {showLabels && <>
        <text x={V[0][0]} y={V[0][1] - 12} textAnchor="middle" fill={CLS.F} fontSize="15" fontWeight="700">FAVOR</text>
        <text x={V[1][0] - 6} y={V[1][1] + 22} textAnchor="middle" fill={CLS.A} fontSize="15" fontWeight="700">AGAINST</text>
        <text x={V[2][0] + 6} y={V[2][1] + 22} textAnchor="middle" fill={CLS.N} fontSize="15" fontWeight="700">NONE</text></>}
      {poly && <polygon points={poly.map(v => v.join(",")).join(" ")} fill={fill} fillOpacity=".28" stroke={fill} strokeWidth="2.5" strokeLinejoin="round" />}
      {points.map((p, i) => { const q = bary(p, V); return <circle key={i} cx={q[0]} cy={q[1]} r="4" fill={INK} opacity=".85" />; })}
      {point && (() => { const q = bary(point, V); return <g><circle cx={q[0]} cy={q[1]} r="7" fill={GOLD} className="glow" /><circle cx={q[0]} cy={q[1]} r="11" fill="none" stroke={GOLD} strokeWidth="1.5" opacity=".6" /></g>; })()}
      {label && <text x={cx} y={cy + r + 52} textAnchor="middle" fill={INK} fontSize="18" fontWeight="700">{label}</text>}
      {extra}
    </g>
  );
}

// ---------- Slide 1: two situations, same vector ----------
function SameVector() {
  const bars = (x, y, p, scale = 150) => (
    <g>
      {["F", "A", "N"].map((k, i) => <g key={k}>
        <rect x={x} y={y + i * 30} width={p[i] * scale} height="20" rx="5" fill={CLS[k]} />
        <text x={x - 10} y={y + i * 30 + 15} textAnchor="end" fill={MUTED} fontSize="13" className="mono">{k === "F" ? "FAVOR" : k === "A" ? "AGAINST" : "NONE"}</text>
        <text x={x + p[i] * scale + 8} y={y + i * 30 + 15} fill={INK} fontSize="13" className="mono">{p[i].toFixed(2)}</text>
      </g>)}
    </g>
  );
  const p = [0.18, 0.62, 0.20];
  return (
    <svg viewBox="0 0 1400 440">
      {/* left card: ambiguity */}
      <rect x="20" y="30" width="520" height="150" rx="16" fill="rgba(255,196,107,.07)" stroke="rgba(255,196,107,.45)" />
      <text x="44" y="62" fill={GOLD} fontSize="14" className="mono" fontWeight="700">AMBIGUITY · annotators disagree</text>
      <text x="44" y="96" fill={INK} fontSize="19" fontStyle="italic">“Oh great, another ‘climate emergency’ summit.</text>
      <text x="44" y="124" fill={INK} fontSize="19" fontStyle="italic">I’m sure the private jets will fix it.”</text>
      <text x="44" y="160" fill={MUTED} fontSize="14">target: Climate Change · 8 votes: 5 AGAINST · 2 FAVOR · 1 NONE</text>
      {/* right card: ignorance */}
      <rect x="20" y="260" width="520" height="150" rx="16" fill="rgba(255,111,156,.07)" stroke="rgba(255,111,156,.45)" />
      <text x="44" y="292" fill={ROSE} fontSize="14" className="mono" fontWeight="700">IGNORANCE · target never seen in training</text>
      <text x="44" y="326" fill={INK} fontSize="19" fontStyle="italic">“The digital euro pilot starts next spring.</text>
      <text x="44" y="354" fill={INK} fontSize="19" fontStyle="italic">Banks are already lobbying against the caps.”</text>
      <text x="44" y="390" fill={MUTED} fontSize="14">target: Digital euro · training covered abortion, feminism, atheism…</text>
      {/* arrows into the model */}
      <path d="M545 105 C 640 105, 640 220, 700 220" fill="none" stroke={DIM} strokeWidth="2.5" className="flow" />
      <path d="M545 335 C 640 335, 640 220, 700 220" fill="none" stroke={DIM} strokeWidth="2.5" className="flow" />
      <rect x="700" y="170" width="190" height="100" rx="16" fill="rgba(139,151,255,.12)" stroke={ACC} strokeWidth="1.5" />
      <text x="795" y="212" textAnchor="middle" fill={INK} fontSize="17" fontWeight="700">stance model</text>
      <text x="795" y="238" textAnchor="middle" fill={MUTED} fontSize="14" className="mono">softmax</text>
      <path d="M895 220 L 960 220" stroke={DIM} strokeWidth="2.5" className="flow" />
      <polygon points="960,213 972,220 960,227" fill={DIM} />
      {/* same vector */}
      <rect x="990" y="140" width="380" height="160" rx="16" fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.15)" />
      <text x="1010" y="170" fill={MUTED} fontSize="13" className="mono">ONE probability vector, for BOTH</text>
      {bars(1085, 188, p, 190)}
      <text x="1180" y="350" textAnchor="middle" fill={ROSE} fontSize="20" fontWeight="800">same output → no basis to decide</text>
      <text x="1180" y="378" textAnchor="middle" fill={MUTED} fontSize="16">“flag as contested” vs “defer, collect labels”</text>
    </svg>
  );
}

// ---------- Credal sets 101 ----------
function CredalIntro() {
  const members = [[0.12, 0.70, 0.18], [0.30, 0.45, 0.25], [0.18, 0.60, 0.22], [0.25, 0.62, 0.13], [0.10, 0.56, 0.34]];
  const l = [0.10, 0.45, 0.13], u = [0.30, 0.70, 0.34];
  return (
    <svg viewBox="0 0 1400 520">
      <Simplex cx={250} cy={260} r={190} point={[0.18, 0.62, 0.20]} label="softmax: a point p ∈ Δ₃" />
      <path d="M470 260 L 560 260" stroke={DIM} strokeWidth="2.5" className="flow" /><polygon points="560,253 572,260 560,267" fill={DIM} />
      <Simplex cx={790} cy={260} r={190} l={l} u={u} points={members} label="credal set 𝒞(l,u): a box of candidate p" />
      {/* bounds panel */}
      <g transform="translate(1040,70)">
        <text x="0" y="0" fill={MUTED} fontSize="14" className="mono">CLASS-WISE INTERVALS [l, u]</text>
        {[["FAVOR", l[0], u[0], CLS.F], ["AGAINST", l[1], u[1], CLS.A], ["NONE", l[2], u[2], CLS.N]].map(([n, a, b, c], i) => (
          <g key={n} transform={`translate(40,${40 + i * 70})`}>
            <text x="0" y="0" fill={c} fontSize="15" fontWeight="700">{n}</text>
            <rect x="0" y="12" width="280" height="18" rx="6" fill="rgba(255,255,255,.06)" />
            <rect x={a * 280} y="12" width={(b - a) * 280} height="18" rx="6" fill={c} opacity=".85" />
            <text x={a * 280 - 6} y="26" fill={MUTED} fontSize="13" className="mono" textAnchor="end">{a.toFixed(2)}</text>
            <text x={b * 280 + 6} y="26" fill={MUTED} fontSize="13" className="mono" textAnchor="start">{b.toFixed(2)}</text>
          </g>))}
        <text x="0" y="270" fill={INK} fontSize="16">lower probability  l<tspan baselineShift="sub" fontSize="11">k</tspan> = min over members</text>
        <text x="0" y="298" fill={INK} fontSize="16">upper probability u<tspan baselineShift="sub" fontSize="11">k</tspan> = max over members</text>
        <text x="0" y="340" fill={MUTED} fontSize="15">white dots = ensemble members p₁…p₅</text>
        <text x="0" y="364" fill={MUTED} fontSize="15">gold dot = point prediction (intersection probability)</text>
      </g>
    </svg>
  );
}

// ---------- Ambiguity vs ignorance via set location and size ----------
function AmbigVsIgnor() {
  return (
    <svg viewBox="0 0 1400 560">
      <Simplex cx={300} cy={250} r={185} l={[0.30, 0.32, 0.22]} u={[0.40, 0.44, 0.34]} fill={GOLD} point={[0.35, 0.38, 0.27]}
        label="ambiguity: small set, centred" />
      <text x="300" y="528" textAnchor="middle" fill={MUTED} fontSize="16">high AU (every p inside is spread out) · low EU</text>
      <Simplex cx={1000} cy={250} r={185} l={[0.05, 0.10, 0.05]} u={[0.60, 0.70, 0.55]} fill={ROSE} point={[0.33, 0.37, 0.30]}
        label="ignorance: large set" />
      <text x="1000" y="528" textAnchor="middle" fill={MUTED} fontSize="16">high EU (we do not know which p is right)</text>
      <text x="650" y="120" textAnchor="middle" fill={INK} fontSize="20" fontWeight="800">same point prediction</text>
      <text x="650" y="150" textAnchor="middle" fill={MUTED} fontSize="16">different sets</text>
      <path d="M560 250 C 600 250, 620 250, 740 250" stroke={DIM} strokeWidth="2" strokeDasharray="5 6" fill="none" />
      <text x="650" y="238" textAnchor="middle" fill={GOLD} fontSize="14" className="mono">location ← ambiguity</text>
      <text x="650" y="285" textAnchor="middle" fill={ROSE} fontSize="14" className="mono">size ← ignorance</text>
    </svg>
  );
}

// ---------- Classical credal networks vs modern credal deep learning ----------
function ClassicalVsModern() {
  const node = (x, y, t) => <g><circle cx={x} cy={y} r="26" fill="rgba(255,255,255,.05)" stroke={MUTED} strokeWidth="1.5" /><text x={x} y={y + 5} textAnchor="middle" fill={INK} fontSize="15" fontWeight="700">{t}</text></g>;
  const edge = (a, b) => <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={MUTED} strokeWidth="1.5" markerEnd="url(#arr)" />;
  return (
    <svg viewBox="0 0 1400 480">
      <defs><marker id="arr" markerWidth="10" markerHeight="10" refX="30" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={MUTED} /></marker></defs>
      {/* classical */}
      <rect x="20" y="20" width="640" height="440" rx="20" fill="rgba(255,255,255,.03)" stroke="rgba(255,255,255,.12)" />
      <text x="44" y="56" fill={MUTED} fontSize="14" className="mono">CLASSICAL · credal networks (Cozman 2000)</text>
      {edge([150, 150], [260, 250])}{edge([370, 150], [260, 250])}{edge([260, 250], [260, 360])}
      {node(150, 150, "A")}{node(370, 150, "B")}{node(260, 250, "C")}{node(260, 360, "Y")}
      <rect x="430" y="110" width="200" height="110" rx="12" fill="rgba(255,255,255,.04)" stroke="rgba(255,255,255,.12)" />
      <text x="530" y="136" textAnchor="middle" fill={MUTED} fontSize="12" className="mono">P(C | A,B) ∈</text>
      <text x="530" y="165" textAnchor="middle" fill={INK} fontSize="16" className="mono">[0.30, 0.45]</text>
      <text x="530" y="192" textAnchor="middle" fill={INK} fontSize="16" className="mono">[0.55, 0.70]</text>
      <text x="44" y="420" fill={INK} fontSize="17">set lives <tspan fontWeight="800">inside</tspan> a graphical model over discrete variables</text>
      <text x="44" y="446" fill={ROSE} fontSize="16">→ exact inference is exponential; tabular inputs only</text>
      {/* modern */}
      <rect x="720" y="20" width="660" height="440" rx="20" fill="rgba(61,214,198,.05)" stroke="rgba(61,214,198,.4)" />
      <text x="744" y="56" fill={ACC2} fontSize="14" className="mono">MODERN · credal deep learning (2024–2026)</text>
      <rect x="744" y="110" width="150" height="70" rx="12" fill="rgba(255,255,255,.05)" stroke={MUTED} /><text x="819" y="140" textAnchor="middle" fill={INK} fontSize="15" fontWeight="700">text + target</text><text x="819" y="162" textAnchor="middle" fill={MUTED} fontSize="12" className="mono">raw tokens</text>
      <path d="M894 145 L 940 145" stroke={ACC2} strokeWidth="2.5" className="flow" />
      <rect x="940" y="95" width="170" height="100" rx="12" fill="rgba(139,151,255,.12)" stroke={ACC} /><text x="1025" y="138" textAnchor="middle" fill={INK} fontSize="15" fontWeight="700">neural encoder</text><text x="1025" y="162" textAnchor="middle" fill={MUTED} fontSize="12" className="mono">DeBERTa / LLM</text>
      <path d="M1110 145 L 1160 145" stroke={ACC2} strokeWidth="2.5" className="flow" />
      <g transform="translate(1160,100)">
        {[["F", 0.15, 0.3, CLS.F], ["A", 0.5, 0.7, CLS.A], ["N", 0.1, 0.25, CLS.N]].map(([k, a, b, c], i) => (
          <g key={k} transform={`translate(0,${i * 30})`}><rect x="0" y="0" width="190" height="18" rx="5" fill="rgba(255,255,255,.06)" /><rect x={a * 190} y="0" width={(b - a) * 190} height="18" rx="5" fill={c} /></g>))}
        <text x="95" y="118" textAnchor="middle" fill={MUTED} fontSize="12" className="mono">credal set at the OUTPUT</text>
      </g>
      <text x="744" y="300" fill={INK} fontSize="17">the set is built from probability vectors the encoder produces</text>
      <text x="744" y="330" fill={INK} fontSize="17">→ any input a neural network accepts: text, images, audio</text>
      <text x="744" y="380" fill={MUTED} fontSize="15">already evaluated on text: ChaosNLI (Javanmardi 2024; Hofman 2026),</text>
      <text x="744" y="404" fill={MUTED} fontSize="15">credal LLMs via LoRA ensembles (Manchingal 2026)</text>
      <text x="744" y="446" fill={ACC2} fontSize="16" fontWeight="700">stance = (text, target) pair, 3 classes, high disagreement — same shape as NLI</text>
    </svg>
  );
}

// ---------- The main system architecture ----------
function Pipeline({step = 99}) {
  const on = (i) => ({opacity: step >= i ? 1 : 0.12, transition: "opacity .4s"});
  const memberColors = [CLS.N, ACC2, GOLD, ACC, VIO];
  const memberNames = ["CE (hard labels)", "Soft (λₓ)", "DRO (top-δ)", "Soft + DRO ★", "CreNet (interval head)"];
  const box = (x, y, w, h, title, sub, stroke = MUTED, fill = "rgba(255,255,255,.04)") => (
    <g><rect x={x} y={y} width={w} height={h} rx="14" fill={fill} stroke={stroke} strokeWidth="1.5" />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 4 : h / 2 + 6)} textAnchor="middle" fill={INK} fontSize="17" fontWeight="700">{title}</text>
      {sub && <text x={x + w / 2} y={y + h / 2 + 20} textAnchor="middle" fill={MUTED} fontSize="13" className="mono">{sub}</text>}</g>);
  const arrow = (d, c = DIM) => <path d={d} fill="none" stroke={c} strokeWidth="2.5" className="flow" />;
  return (
    <svg viewBox="0 0 1580 600">
      <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill={DIM} /></marker></defs>
      {/* stage 1 input + encoder */}
      <g style={on(0)}>
        {box(20, 250, 170, 90, "(text, target)", "“[CLS] text [SEP] target”")}
        {arrow("M190 295 L 235 295")}
        {box(235, 235, 190, 120, "DeBERTa-v3-base", "shared recipe · 5 epochs", ACC, "rgba(139,151,255,.1)")}
        <text x="330" y="380" textAnchor="middle" fill={DIM} fontSize="12" className="mono">seed i + 80% bootstrap</text>
      </g>
      {/* stage 2 members */}
      <g style={on(1)}>
        {memberNames.map((n, i) => { const y = 70 + i * 95; return (<g key={n}>
          {arrow(`M425 295 C 470 295, 470 ${y + 30}, 505 ${y + 30}`, memberColors[i])}
          <rect x="505" y={y} width="230" height="60" rx="12" fill="rgba(255,255,255,.04)" stroke={memberColors[i]} strokeWidth="1.5" />
          <text x="620" y={y + 26} textAnchor="middle" fill={INK} fontSize="15" fontWeight="700">member {i + 1}</text>
          <text x="620" y={y + 46} textAnchor="middle" fill={memberColors[i]} fontSize="12.5" className="mono">{n}</text>
          {arrow(`M735 ${y + 30} C 780 ${y + 30}, 780 295, 820 295`, memberColors[i])}
        </g>); })}
        <text x="620" y="545" textAnchor="middle" fill={MUTED} fontSize="13">five member types compared; one type per system, M = 5</text>
      </g>
      {/* stage 3 credal layer */}
      <g style={on(2)}>
        {box(820, 225, 230, 140, "credal layer", "l = min p · u = max p", ACC2, "rgba(61,214,198,.1)")}
        <text x="935" y="340" textAnchor="middle" fill={MUTED} fontSize="12.5" className="mono">→ reachable(l, u)</text>
        <text x="935" y="572" textAnchor="middle" fill={MUTED} fontSize="13">single-model variants (EffCre, MC-dropout, LLM prompts)</text>
        <text x="935" y="591" textAnchor="middle" fill={MUTED} fontSize="13">plug in at the credal layer</text>
      </g>
      {/* stage 4 three heads */}
      <g style={on(3)}>
        {arrow("M1050 295 C 1090 295, 1090 110, 1130 110", GOLD)}
        {arrow("M1050 295 L 1130 295", ACC)}
        {arrow("M1050 295 C 1090 295, 1090 480, 1130 480", ROSE)}
        {box(1130, 55, 230, 110, "decomposition", "TU = H̄ · AU = H̲ · width", GOLD)}
        {box(1130, 240, 230, 110, "conformal calibration", "q̂ on cal split: box ⊕ q̂", ACC)}
        {box(1130, 425, 230, 110, "interval dominance", "ND(x) = {k : uₖ ≥ max lⱼ}", ROSE)}
      </g>
      <g style={on(4)}>
        <text x="1385" y="95" fill={GOLD} fontSize="14" fontWeight="700">ambiguity vs</text><text x="1385" y="114" fill={GOLD} fontSize="14" fontWeight="700">novelty scores</text>
        <text x="1385" y="282" fill={ACC} fontSize="14" fontWeight="700">90 % coverage of</text><text x="1385" y="301" fill={ACC} fontSize="14" fontWeight="700">human λₓ</text>
        <text x="1385" y="466" fill={ROSE} fontSize="14" fontWeight="700">predict / set /</text><text x="1385" y="485" fill={ROSE} fontSize="14" fontWeight="700">defer</text>
      </g>
    </svg>
  );
}

// ---------- Member training objectives (mini SVGs) ----------
function MemberMini({kind}) {
  const W = 300, H = 150;
  if (kind === "ce") return (<svg viewBox={`0 0 ${W} ${H}`}>
    <text x="10" y="20" fill={MUTED} fontSize="12" className="mono">target: one-hot y</text>
    {["F", "A", "N"].map((k, i) => <g key={k}><rect x="90" y={38 + i * 32} width="190" height="20" rx="5" fill="rgba(255,255,255,.06)" /><rect x="90" y={38 + i * 32} width={i === 1 ? 190 : 0} height="20" rx="5" fill={CLS[k]} /><text x="80" y={53 + i * 32} textAnchor="end" fill={CLS[k]} fontSize="12" className="mono">{k}</text></g>)}
  </svg>);
  if (kind === "soft") return (<svg viewBox={`0 0 ${W} ${H}`}>
    <text x="10" y="20" fill={MUTED} fontSize="12" className="mono">target: human votes λₓ = (2,5,1)/8</text>
    {[[0.25, "F"], [0.625, "A"], [0.125, "N"]].map(([v, k], i) => <g key={k}><rect x="90" y={38 + i * 32} width="190" height="20" rx="5" fill="rgba(255,255,255,.06)" /><rect x="90" y={38 + i * 32} width={190 * v} height="20" rx="5" fill={CLS[k]} /><text x="80" y={53 + i * 32} textAnchor="end" fill={CLS[k]} fontSize="12" className="mono">{k}</text><text x={95 + 190 * v} y={53 + i * 32} fill={INK} fontSize="12" className="mono">{v}</text></g>)}
  </svg>);
  if (kind === "dro") { const losses = [2.6, 2.1, 1.7, 1.4, 1.1, 0.9, 0.7, 0.5, 0.35, 0.2]; const k = 5; return (<svg viewBox={`0 0 ${W} ${H}`}>
    <text x="10" y="20" fill={MUTED} fontSize="12" className="mono">batch sorted by loss · keep top δ·B</text>
    {losses.map((v, i) => <rect key={i} x={20 + i * 27} y={130 - v * 36} width="20" height={v * 36} rx="3" fill={i < k ? GOLD : "rgba(255,255,255,.12)"} />)}
    <line x1={20 + k * 27 - 4} x2={20 + k * 27 - 4} y1="35" y2="135" stroke={GOLD} strokeDasharray="4 4" />
    <text x={20 + k * 27 + 2} y="48" fill={GOLD} fontSize="12" className="mono">δ = 0.5</text>
    <text x="150" y="145" fill={DIM} fontSize="11" className="mono">gradient only from the hard tail</text>
  </svg>); }
  return (<svg viewBox={`0 0 ${W} ${H}`}>
    <text x="10" y="20" fill={MUTED} fontSize="12" className="mono">head outputs m and h = softplus(·)</text>
    {[[0.2, 0.08, "F"], [0.6, 0.1, "A"], [0.2, 0.06, "N"]].map(([m, h, k], i) => <g key={k}>
      <rect x="90" y={38 + i * 32} width="190" height="20" rx="5" fill="rgba(255,255,255,.06)" />
      <rect x={90 + 190 * (m - h)} y={38 + i * 32} width={190 * 2 * h} height="20" rx="5" fill={CLS[k]} opacity=".9" />
      <line x1={90 + 190 * m} x2={90 + 190 * m} y1={36 + i * 32} y2={60 + i * 32} stroke={INK} strokeWidth="2" />
      <text x="80" y={53 + i * 32} textAnchor="end" fill={CLS[k]} fontSize="12" className="mono">{k}</text></g>)}
    <text x="90" y="145" fill={DIM} fontSize="11" className="mono">[m−h, m+h] → interval softmax → [q_L, q_U]</text>
  </svg>);
}

// ---------- The proposal: CreDRO-Soft ----------
function CreDROSoftDiagram({step = 99}) {
  const on = (i) => ({opacity: step >= i ? 1 : 0.12, transition: "opacity .4s"});
  const votes = ["A", "A", "F", "A", "N", "A", "F", "A"];
  const deltas = [0.5, 0.625, 0.75, 0.875, 1.0];
  return (
    <svg viewBox="0 0 1500 560">
      {/* left: votes → λ */}
      <g style={on(0)}>
        <text x="30" y="40" fill={ACC2} fontSize="14" className="mono">① HUMAN LABEL DISTRIBUTION (no new annotation)</text>
        {votes.map((v, i) => <g key={i} transform={`translate(${40 + (i % 4) * 70},${70 + Math.floor(i / 4) * 70})`}>
          <circle cx="24" cy="18" r="11" fill={CLS[v]} opacity=".9" /><rect x="8" y="32" width="32" height="24" rx="8" fill={CLS[v]} opacity=".9" />
          <text x="24" y="70" textAnchor="middle" fill={CLS[v]} fontSize="11" className="mono">{v === "A" ? "AGAINST" : v === "F" ? "FAVOR" : "NONE"}</text></g>)}
        <text x="40" y="235" fill={MUTED} fontSize="14">8 SemEval-2016 annotators per tweet</text>
        <path d="M330 140 L 380 140" stroke={ACC2} strokeWidth="2.5" className="flow" />
        <g transform="translate(390,70)">
          {[[0.25, "F"], [0.625, "A"], [0.125, "N"]].map(([v, k], i) => <g key={k}><rect x="0" y={i * 36} width="200" height="24" rx="6" fill="rgba(255,255,255,.06)" /><rect x="0" y={i * 36} width={200 * v} height="24" rx="6" fill={CLS[k]} /><text x={200 * v + 8} y={i * 36 + 17} fill={INK} fontSize="13" className="mono">{v}</text></g>)}
          <text x="0" y="135" fill={INK} fontSize="15" fontWeight="700">λₓ = (n_F, n_A, n_N) / n</text>
          <text x="0" y="158" fill={MUTED} fontSize="13">per-sample loss: −Σₖ λₓ,ₖ log pₖ</text>
        </g>
      </g>
      {/* middle: DRO members */}
      <g style={on(1)}>
        <text x="30" y="300" fill={GOLD} fontSize="14" className="mono">② MEMBER DIVERSITY BY DESIGN (CVaR-DRO, δᵢ ∈ [0.5, 1])</text>
        {deltas.map((d, i) => { const x = 40 + i * 125; const losses = [2.5, 2.0, 1.6, 1.3, 1.0, 0.8, 0.6, 0.4]; const k = Math.round(d * 8); return (
          <g key={d} transform={`translate(${x},320)`}>
            <rect x="0" y="0" width="110" height="150" rx="12" fill="rgba(255,255,255,.03)" stroke={ACC} strokeOpacity=".5" />
            <text x="55" y="22" textAnchor="middle" fill={INK} fontSize="13" fontWeight="700">member {i + 1}</text>
            {losses.map((v, j) => <rect key={j} x={10 + j * 11.5} y={120 - v * 30} width="9" height={v * 30} rx="2" fill={j < k ? GOLD : "rgba(255,255,255,.12)"} />)}
            <text x="55" y="142" textAnchor="middle" fill={GOLD} fontSize="12" className="mono">δ = {d}</text>
          </g>); })}
        <text x="40" y="500" fill={MUTED} fontSize="14">δ = 0.5 trains on the hardest half only · δ = 1 trains on everything</text>
        <text x="40" y="522" fill={MUTED} fontSize="14">members diverge most on items unlike the training distribution — exactly what an unseen target looks like</text>
      </g>
      {/* right: the credal set */}
      <g style={on(2)}>
        <Simplex cx={1120} cy={250} r={190} l={[0.12, 0.50, 0.08]} u={[0.38, 0.72, 0.28]} points={[[0.25, 0.6, 0.15], [0.3, 0.55, 0.15], [0.2, 0.62, 0.18], [0.22, 0.68, 0.10], [0.33, 0.52, 0.15]]} point={[0.26, 0.60, 0.14]} />
        <text x="1120" y="500" textAnchor="middle" fill={INK} fontSize="19" fontWeight="800">CreDRO-Soft credal set</text>
        <path d="M640 110 C 800 110, 900 160, 1010 200" fill="none" stroke={ACC2} strokeWidth="2.5" className="flow" />
        <text x="830" y="100" textAnchor="middle" fill={ACC2} fontSize="15" fontWeight="700">location ← human disagreement</text>
        <path d="M690 380 C 820 380, 900 330, 980 300" fill="none" stroke={GOLD} strokeWidth="2.5" className="flow" />
        <text x="830" y="410" textAnchor="middle" fill={GOLD} fontSize="15" fontWeight="700">size ← distributional robustness</text>
      </g>
    </svg>
  );
}

// ---------- Interval dominance + routing ----------
function Dominance() {
  const rows = [
    {t: "clear case", l: [0.05, 0.70, 0.05], u: [0.15, 0.85, 0.15], out: "predict AGAINST", c: ACC2},
    {t: "contested", l: [0.30, 0.35, 0.05], u: [0.50, 0.55, 0.15], out: "set {FAVOR, AGAINST}", c: GOLD},
    {t: "unseen target", l: [0.05, 0.10, 0.05], u: [0.60, 0.70, 0.55], out: "defer to human", c: ROSE},
  ];
  return (
    <svg viewBox="0 0 1000 440">
      {rows.map((r, i) => { const y = 30 + i * 135; const maxl = Math.max(...r.l); return (
        <g key={r.t} transform={`translate(0,${y})`}>
          <text x="10" y="50" fill={INK} fontSize="17" fontWeight="700">{r.t}</text>
          {["F", "A", "N"].map((k, j) => { const x0 = 180; const sc = 380; const nd = r.u[j] >= maxl - 1e-9; return (<g key={k}>
            <text x={x0 - 12} y={22 + j * 30 + 14} textAnchor="end" fill={CLS[k]} fontSize="12" className="mono">{k}</text>
            <rect x={x0} y={22 + j * 30} width={sc} height="18" rx="5" fill="rgba(255,255,255,.05)" />
            <rect x={x0 + r.l[j] * sc} y={22 + j * 30} width={(r.u[j] - r.l[j]) * sc} height="18" rx="5" fill={CLS[k]} opacity={nd ? .95 : .3} />
          </g>); })}
          <line x1={180 + maxl * 380} x2={180 + maxl * 380} y1="14" y2="112" stroke={INK} strokeDasharray="4 4" strokeWidth="1.5" />
          <text x={180 + maxl * 380 + 6} y="12" fill={MUTED} fontSize="11" className="mono">max lⱼ</text>
          <text x="590" y="66" fill={MUTED} fontSize="14" className="mono">ND = {`{${["F", "A", "N"].filter((k, j) => r.u[j] >= maxl - 1e-9).join(", ")}}`}</text>
          <path d="M700 60 L 740 60" stroke={r.c} strokeWidth="2.5" className="flow" />
          <rect x="745" y="36" width="240" height="50" rx="12" fill={r.c} fillOpacity=".12" stroke={r.c} />
          <text x="865" y="67" textAnchor="middle" fill={r.c} fontSize="16" fontWeight="800">{r.out}</text>
        </g>); })}
    </svg>
  );
}

// ---------- Conformal credal calibration (box dilation) ----------
function ConformalDiag() {
  return (
    <svg viewBox="0 0 700 460">
      <Simplex cx={230} cy={230} r={185} l={[0.10, 0.50, 0.10]} u={[0.30, 0.70, 0.30]} point={[0.2, 0.6, 0.2]} />
      {(() => { const V = simplexVerts(230, 230, 185); const q = 0.09; const poly = credalPolygon([0.10 - q, 0.50 - q, 0.10 - q], [0.30 + q, 0.70 + q, 0.30 + q]).map(p => bary(p, V));
        const lam = bary([0.33, 0.52, 0.15], V);
        return <g><polygon points={poly.map(v => v.join(",")).join(" ")} fill="none" stroke={ACC2} strokeWidth="2.5" strokeDasharray="7 6" />
          <circle cx={lam[0]} cy={lam[1]} r="6" fill={CLS.F} stroke={INK} strokeWidth="1.5" />
          <text x={lam[0] + 12} y={lam[1] + 4} fill={INK} fontSize="14" fontWeight="700">λₓ (human)</text></g>; })()}
      <g transform="translate(470,60)">
        <text x="0" y="0" fill={MUTED} fontSize="13" className="mono">ON THE CAL SPLIT</text>
        <text x="0" y="30" fill={INK} fontSize="15">score sᵢ = dist(λᵢ, box(xᵢ))</text>
        <text x="0" y="55" fill={INK} fontSize="15">q̂ = ⌈(n+1)(1−α)⌉/n quantile</text>
        <text x="0" y="100" fill={MUTED} fontSize="13" className="mono">AT TEST TIME</text>
        <text x="0" y="130" fill={INK} fontSize="15">dilate box by q̂ (dashed)</text>
        <text x="0" y="170" fill={ACC2} fontSize="15" fontWeight="700">P(λₓ ∈ set) ≥ 1 − α</text>
        <text x="0" y="195" fill={MUTED} fontSize="13">distribution-free, under exchangeability</text>
        <text x="0" y="240" fill={MUTED} fontSize="13">smaller q̂ = model box already</text>
        <text x="0" y="260" fill={MUTED} fontSize="13">close to human distribution</text>
      </g>
    </svg>
  );
}

// ---------- LOTO folds ----------
function LOTO() {
  const T = ["Atheism", "Climate", "Feminist", "Hillary", "Abortion"];
  return (
    <svg viewBox="0 0 700 330">
      {T.map((t, j) => <text key={t} x={150 + j * 100} y="40" textAnchor="middle" fill={MUTED} fontSize="13" fontWeight="700">{t}</text>)}
      {T.map((t, i) => <g key={t}>
        <text x="90" y={80 + i * 46} textAnchor="end" fill={INK} fontSize="13" className="mono">fold {i + 1}</text>
        {T.map((_, j) => <rect key={j} x={110 + j * 100} y={60 + i * 46} width="80" height="32" rx="8" fill={i === j ? ROSE : ACC2} fillOpacity={i === j ? .85 : .35} stroke={i === j ? ROSE : "none"} />)}
        {T.map((_, j) => <text key={"t" + j} x={150 + j * 100} y={81 + i * 46} textAnchor="middle" fill={i === j ? "#1a0a10" : INK} fontSize="11" className="mono" fontWeight="700">{i === j ? "UNSEEN" : "train"}</text>)}
      </g>)}
      <rect x="110" y="300" width="14" height="14" rx="4" fill={ACC2} fillOpacity=".35" /><text x="132" y="312" fill={MUTED} fontSize="13">trained on (4 targets)</text>
      <rect x="330" y="300" width="14" height="14" rx="4" fill={ROSE} fillOpacity=".85" /><text x="352" y="312" fill={MUTED} fontSize="13">held out → test items of that target are “unseen”</text>
    </svg>
  );
}

// ---------- Entropy decomposition ----------
function DecompDiag() {
  return (
    <svg viewBox="0 0 800 460">
      <Simplex cx={230} cy={230} r={185} l={[0.10, 0.45, 0.10]} u={[0.40, 0.75, 0.30]} point={[0.333, 0.334, 0.333]} fill={ACC}
        extra={(() => { const V = simplexVerts(230, 230, 185); const mn = bary([0.10, 0.75, 0.15], V); const mx = bary([0.30, 0.45, 0.25], V); return <g>
          <circle cx={mn[0]} cy={mn[1]} r="7" fill={ROSE} /><text x={mn[0] + 12} y={mn[1] + 5} fill={ROSE} fontSize="14" fontWeight="700">argmin H → H̲ (AU)</text>
          <circle cx={mx[0]} cy={mx[1]} r="7" fill={GOLD} /><text x={mx[0] + 12} y={mx[1] - 8} fill={GOLD} fontSize="14" fontWeight="700">argmax H → H̄ (TU)</text></g>; })()} />
      <g transform="translate(470,70)">
        <text x="0" y="0" fill={MUTED} fontSize="13" className="mono">ENTROPY IN BITS</text>
        <rect x="0" y="20" width="200" height="26" rx="7" fill="rgba(255,255,255,.06)" />
        <rect x="0" y="20" width="92" height="26" rx="7" fill={ROSE} /><rect x="92" y="20" width="78" height="26" fill={GOLD} />
        <text x="46" y="38" textAnchor="middle" fill="#1a0a10" fontSize="12" fontWeight="800">AU = H̲</text>
        <text x="131" y="38" textAnchor="middle" fill="#1a0a10" fontSize="12" fontWeight="800">EU</text>
        <text x="0" y="68" fill={MUTED} fontSize="13" className="mono">|←——— TU = H̄ ———→|</text>
        <text x="0" y="120" fill={INK} fontSize="15">H̄ = entropy of the most spread-out p in 𝒞</text>
        <text x="0" y="146" fill={INK} fontSize="15">H̲ = entropy of the most peaked p in 𝒞</text>
        <text x="0" y="172" fill={INK} fontSize="15">EU = H̄ − H̲  (Abellán et al. 2006)</text>
        <text x="0" y="215" fill={ACC2} fontSize="15" fontWeight="700">also used: width = Σₖ (uₖ − lₖ)</text>
        <text x="0" y="240" fill={MUTED} fontSize="13">a size proxy that does not depend on entropy</text>
        <text x="0" y="290" fill={MUTED} fontSize="13">H̄: water-filling bisection</text>
        <text x="0" y="310" fill={MUTED} fontSize="13">H̲: vertex enumeration of the box</text>
      </g>
    </svg>
  );
}

// ---------- EU pathology on unseen targets ----------
function EUPathology() {
  return (
    <svg viewBox="0 0 1400 460">
      <Simplex cx={260} cy={200} r={150} l={[0.08, 0.55, 0.05]} u={[0.30, 0.80, 0.25]} point={[0.18, 0.66, 0.16]} fill={ACC} label="seen target" />
      <text x="260" y="435" textAnchor="middle" fill={MUTED} fontSize="14">H̲ small, H̄ moderate → EU sizeable</text>
      <path d="M460 200 L 540 200" stroke={DIM} strokeWidth="2.5" className="flow" /><polygon points="540,193 552,200 540,207" fill={DIM} />
      <Simplex cx={760} cy={200} r={150} l={[0.20, 0.22, 0.18]} u={[0.48, 0.50, 0.44]} point={[0.33, 0.35, 0.32]} fill={ROSE} label="unseen target" />
      <text x="760" y="435" textAnchor="middle" fill={MUTED} fontSize="14">every member near-uniform → H̲ ≈ H̄ ≈ log₂3 → EU shrinks</text>
      <g transform="translate(980,60)">
        <text x="0" y="0" fill={ROSE} fontSize="16" fontWeight="800">entropy EU fails as novelty score</text>
        <text x="0" y="30" fill={MUTED} fontSize="14">LOTO AUROC (EU) for CreDRO-Soft: 0.337</text>
        <text x="0" y="52" fill={MUTED} fontSize="14">mean AU seen→unseen: 0.64 → 1.06 (saturates)</text>
        <text x="0" y="74" fill={MUTED} fontSize="14">mean EU seen→unseen: 0.60 → 0.48 (drops!)</text>
        <text x="0" y="120" fill={ACC2} fontSize="16" fontWeight="800">use upper entropy or width instead</text>
        <text x="0" y="150" fill={MUTED} fontSize="14">LOTO AUROC (TU): 0.905 · (width): 0.759</text>
        <text x="0" y="172" fill={MUTED} fontSize="14">COVID AUROC (width) for CreDE: 0.902</text>
        <text x="0" y="220" fill={DIM} fontSize="13">consistent with Wimmer et al. 2023 and</text>
        <text x="0" y="240" fill={DIM} fontSize="13">Sale et al. 2023 on multiclass credal measures</text>
      </g>
    </svg>
  );
}
