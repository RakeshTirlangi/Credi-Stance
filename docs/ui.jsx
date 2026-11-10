// Shared UI primitives: step-reveal fragments, KaTeX, cards.
const SlideCtx = React.createContext({step: 0});

function Frag({i = 1, children, dim = false, as: Tag = "div", className = "", style}) {
  const {step} = React.useContext(SlideCtx);
  const on = step >= i;
  return <Tag className={`frag ${dim ? "dimmable" : ""} ${on ? "on" : ""} ${className}`} style={style}>{children}</Tag>;
}

function K({tex, block = false, className = ""}) {
  const html = React.useMemo(() => {
    try { return katex.renderToString(tex, {displayMode: block, throwOnError: false}); }
    catch (e) { return tex; }
  }, [tex, block]);
  return <span className={className} dangerouslySetInnerHTML={{__html: html}} />;
}
const Eq = ({tex, small}) => <div className={`eq ${small ? "small" : ""}`}><K tex={tex} block /></div>;

function Card({tag, title, children, className = "", style}) {
  return (
    <div className={`card ${className}`} style={style}>
      {tag && <span className="tag">{tag}</span>}
      {title && <h3>{title}</h3>}
      {children}
    </div>
  );
}

function Header({kicker, title, sub}) {
  return (
    <div>
      {kicker && <div className="kicker">{kicker}</div>}
      <h2>{title}</h2>
      {sub && <p className="sub">{sub}</p>}
    </div>
  );
}

function Step({n, title, children}) {
  return (
    <div className="step">
      <div className="n">{n}</div>
      <div><b style={{fontSize: 21}}>{title}</b><p>{children}</p></div>
    </div>
  );
}

const Pill = ({kind = "", children}) => <span className={`pill ${kind}`}>{children}</span>;
