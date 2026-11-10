// Deck shell: scaling stage, keyboard/touch navigation, hash routing, overview, notes, help.
function useScale() {
  const [s, setS] = React.useState(1);
  React.useEffect(() => {
    const f = () => setS(Math.min(window.innerWidth / 1600, window.innerHeight / 900));
    f(); window.addEventListener("resize", f); return () => window.removeEventListener("resize", f);
  }, []);
  return s;
}
function parseHash() { const m = location.hash.match(/#\/(\d+)(?:\/(\d+))?/); if (!m) return [0, 0]; const k = Math.min(SLIDES.length - 1, +m[1]); return [k, Math.min(+(m[2] || 0), SLIDES[k].steps)]; }

function Deck() {
  const [[i, step], setPos] = React.useState(parseHash);
  const [overview, setOverview] = React.useState(false);
  const [notes, setNotes] = React.useState(false);
  const [help, setHelp] = React.useState(false);
  const scale = useScale();
  const slide = SLIDES[i];

  React.useEffect(() => { location.hash = `#/${i}${step ? "/" + step : ""}`; }, [i, step]);
  React.useEffect(() => { const h = () => setPos(parseHash()); window.addEventListener("hashchange", h); return () => window.removeEventListener("hashchange", h); }, []);

  const next = React.useCallback(() => setPos(([a, s]) => s < SLIDES[a].steps ? [a, s + 1] : a < SLIDES.length - 1 ? [a + 1, 0] : [a, s]), []);
  const prev = React.useCallback(() => setPos(([a, s]) => s > 0 ? [a, s - 1] : a > 0 ? [a - 1, SLIDES[a - 1].steps] : [a, s]), []);
  const go = (k) => { setPos([k, 0]); setOverview(false); };

  React.useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === "INPUT") return;
      const k = e.key;
      if (k === "ArrowRight" || k === " " || k === "PageDown" || k === "ArrowDown") { e.preventDefault(); overview ? go(Math.min(i + 1, SLIDES.length - 1)) : next(); }
      else if (k === "ArrowLeft" || k === "PageUp" || k === "ArrowUp" || k === "Backspace") { e.preventDefault(); overview ? go(Math.max(i - 1, 0)) : prev(); }
      else if (k === "Home") setPos([0, 0]); else if (k === "End") setPos([SLIDES.length - 1, 0]);
      else if (k === "o" || k === "O") setOverview(v => !v);
      else if (k === "n" || k === "N") setNotes(v => !v);
      else if (k === "f" || k === "F") { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); }
      else if (k === "?" || k === "h") setHelp(v => !v);
      else if (k === "Escape") { setOverview(false); setHelp(false); }
      else if (/^[0-9]$/.test(k)) { /* ignore */ }
    };
    window.addEventListener("keydown", onKey); return () => window.removeEventListener("keydown", onKey);
  }, [next, prev, overview, i]);

  // touch swipe
  const tx = React.useRef(null);
  const onTouchStart = (e) => { tx.current = e.touches[0].clientX; };
  const onTouchEnd = (e) => { if (tx.current == null) return; const dx = e.changedTouches[0].clientX - tx.current; if (Math.abs(dx) > 50) (dx < 0 ? next() : prev()); tx.current = null; };

  const progress = ((i + (slide.steps ? step / (slide.steps + 1) : 0)) / (SLIDES.length - 1)) * 100;
  return (
    <div className="deck" onTouchStart={onTouchStart} onTouchEnd={onTouchEnd} onClick={(e) => { if (e.target.closest(".nav,.overview,.help,.notes,a")) return; next(); }}>
      <div className="stage" style={{transform: `translate(-50%,-50%) scale(${scale})`}}>
        <SlideCtx.Provider value={{step}}>
          <React.Fragment key={i}>{slide.render(step)}</React.Fragment>
        </SlideCtx.Provider>
      </div>
      <div className="progress" style={{width: `${progress}%`}} />
      <div className="counter"><b>{String(i + 1).padStart(2, "0")} / {SLIDES.length}</b> · {slide.title}{slide.steps ? ` · step ${step}/${slide.steps}` : ""} · press ? for help</div>
      <div className="nav">
        <button title="overview (O)" onClick={() => setOverview(v => !v)}>▦</button>
        <button title="notes (N)" onClick={() => setNotes(v => !v)}>✎</button>
        <button title="fullscreen (F)" onClick={() => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()}>⛶</button>
        <button title="previous" onClick={prev}>‹</button>
        <button title="next" onClick={next}>›</button>
      </div>
      {notes && <div className="notes"><b>Speaker notes · {slide.title}</b>{slide.notes || "—"}</div>}
      {overview && (
        <div className="overview" onClick={(e) => e.stopPropagation()}>
          <div className="grid">
            {SLIDES.map((s, k) => (
              <div key={k} className={`thumb ${k === i ? "cur" : ""}`} onClick={() => go(k)}>
                <div className="mini" style={{transform: `scale(${(window.innerWidth - 80 - 66) / 4 / 1600})`}}>
                  <div className="deck" style={{position: "absolute"}}><SlideCtx.Provider value={{step: s.steps}}>{s.render(s.steps)}</SlideCtx.Provider></div>
                </div>
                <div className="lbl">{k + 1} · {s.title}</div>
              </div>))}
          </div>
        </div>)}
      {help && (
        <div className="help" onClick={() => setHelp(false)}>
          <div className="card" onClick={(e) => e.stopPropagation()}>
            <h3>Keyboard</h3>
            <p><span>Next step / slide</span><span><kbd>→</kbd> <kbd>Space</kbd> <kbd>click</kbd></span></p>
            <p><span>Previous</span><span><kbd>←</kbd> <kbd>Backspace</kbd></span></p>
            <p><span>First / last</span><span><kbd>Home</kbd> <kbd>End</kbd></span></p>
            <p><span>Slide overview</span><span><kbd>O</kbd></span></p>
            <p><span>Speaker notes</span><span><kbd>N</kbd></span></p>
            <p><span>Fullscreen</span><span><kbd>F</kbd></span></p>
            <p><span>This help</span><span><kbd>?</kbd></span></p>
            <p style={{color: "var(--dim)", fontSize: 14, marginTop: 14}}>Every slide has a URL (#/n), so you can link to one directly.</p>
          </div>
        </div>)}
    </div>
  );
}
ReactDOM.createRoot(document.getElementById("root")).render(<Deck />);
