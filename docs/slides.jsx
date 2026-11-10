// Slide definitions. Each slide: {title, steps, notes, render(step)}
const D = window.DATA;
const rows = (arr, i) => arr.map(r => ({name: r[0], v: r[i]}));
const SLIDES = [
  // 0 ---------------------------------------------------------------- title
  {title: "Title", steps: 0, notes: "Open with the one-line idea: replace the probability vector with a set of probabilities so the model can say what kind of uncertainty it has.",
    render: () => (
      <div className="slide title-slide">
        <div className="kicker">Major project · stance detection · uncertainty quantification</div>
        <h1>Credal Uncertainty Quantification for <span className="grad">Reliable Stance Detection</span> on Unseen Targets</h1>
        <p className="sub" style={{marginTop: 26, fontSize: 26}}>Six ways of giving a stance detector a credal output, and one new combination that learns <b>where</b> the set should sit from human disagreement and <b>how big</b> it should be from distributional robustness.</p>
        <div className="title-meta">
          <Pill kind="acc">DeBERTa-v3-base</Pill><Pill kind="teal">SemEval-2016 · P-Stance · COVID-19</Pill><Pill kind="gold">166 trained members</Pill><Pill>no new annotation</Pill>
        </div>
        <div style={{display: "flex", gap: 70, marginTop: 70}}>
          <div className="hero-num"><div className="v grad">0.154 → 0.041</div><div className="k">expected calibration error, softmax → proposed</div></div>
          <div className="hero-num"><div className="v grad">0.905 ± 0.021</div><div className="k">unseen-target AUROC, 5/5 folds over softmax</div></div>
          <div className="hero-num"><div className="v grad">−25…50 %</div><div className="k">decision cost via set-valued answers</div></div>
        </div>
      </div>)},
  // 1 ---------------------------------------------------------------- motivation
  {title: "Motivation", steps: 1, notes: "Two tweets, same output vector. The first is genuinely contested; the second is about something the model has never seen. The actions differ, but softmax cannot tell them apart.",
    render: () => (
      <div className="slide">
        <Header kicker="Motivation" title="One probability vector hides two very different situations" />
        <div className="body"><div className="svgwrap"><SameVector /></div>
          <Frag i={1}><div className="row" style={{gap: 20}}>
            <Card className="accent" style={{flex: 1}}><p><b>SemEval-2016 dropped 1,385 tweets</b> from the shared task because annotator agreement fell below 60 %. Disagreement is the norm, not noise.</p></Card>
            <Card className="rose" style={{flex: 1}}><p><b>Unseen targets are the deployment reality.</b> Every method in this study reaches only ≈0.5 accuracy on a new target. Knowing <i>when</i> that happens is the problem.</p></Card>
          </div></Frag>
        </div>
      </div>)},
  // 2 ---------------------------------------------------------------- problem statement
  {title: "Problem statement", steps: 4, notes: "Read the statement, then the four research questions. Emphasise the constraints: no new annotation, one GPU.",
    render: () => (
      <div className="slide">
        <Header kicker="Problem definition" title="From a point to a set" />
        <div className="body">
          <div className="quote">Build a stance detector that outputs a <span className="grad">credal set</span> <K tex="\mathcal{C}(x,t)\subseteq\Delta_3" />, given by class-wise bounds <K tex="[\underline{P}(y),\overline{P}(y)]" />, such that its <b>location</b> reflects ambiguity, its <b>size</b> reflects ignorance, it can be <b>calibrated</b> to cover the human label distribution, and it <b>supports</b> predict / set-valued / defer decisions. No new annotation. One GPU.</div>
          <div className="grid4" style={{marginTop: 36}}>
            <Frag i={1}><Card tag="RQ1" title="Better uncertainty?"><p>Do credal models beat softmax, temperature scaling, MC-dropout, deep ensembles and evidential DL on calibration and selective risk at equal accuracy?</p></Card></Frag>
            <Frag i={2}><Card tag="RQ2" title="Ambiguity ≠ ignorance?"><p>Does the epistemic score rise on unseen targets while the aleatoric score tracks human label entropy?</p></Card></Frag>
            <Frag i={3}><Card tag="RQ3" title="Coverage guarantee?"><p>Can conformal calibration make the set cover the human label distribution 90 % of the time, and which construction gives the tightest sets?</p></Card></Frag>
            <Frag i={4}><Card tag="RQ4" title="Decision value?"><p>Does the set-valued answer lower expected decision cost compared with confidence thresholding?</p></Card></Frag>
          </div>
        </div>
      </div>)},
  // 3 ---------------------------------------------------------------- credal sets 101
  {title: "Credal sets", steps: 1, notes: "A credal set is a convex set of distributions. We use probability-interval sets: a box of lower and upper probabilities intersected with the simplex. The ensemble members are the white dots; the box is their min/max envelope.",
    render: () => (
      <div className="slide">
        <Header kicker="Background" title="Credal sets: a box of probabilities instead of a point" />
        <div className="body"><div className="svgwrap" style={{flex: 1}}><CredalIntro /></div>
          <Frag i={1}><div className="grid3">
            <Card><Eq small tex="\mathcal{C}(l,u)=\{p\in\Delta_K:\; l_k\le p_k\le u_k\}" /><p style={{textAlign: "center"}}>probability-interval credal set (Walley 1991)</p></Card>
            <Card><Eq small tex="p^*_k = l_k+\alpha\,(u_k-l_k),\quad \alpha=\tfrac{1-\sum_j l_j}{\sum_j (u_j-l_j)}" /><p style={{textAlign: "center"}}>intersection probability: the point prediction</p></Card>
            <Card><Eq small tex="l_k \leftarrow \max\big(l_k,\,1-\textstyle\sum_{j\ne k}u_j\big)" /><p style={{textAlign: "center"}}>reachability: tighten bounds that no p attains</p></Card>
          </div></Frag>
        </div>
      </div>)},
  // 4 ---------------------------------------------------------------- two kinds of uncertainty
  {title: "Two kinds of uncertainty", steps: 0, notes: "Location versus size. Left: small set near the centre = confident that the item is ambiguous. Right: large set = do not know which distribution is right. Same point prediction in both.",
    render: () => (
      <div className="slide">
        <Header kicker="Background" title="Location encodes ambiguity, size encodes ignorance" />
        <div className="body"><div className="svgwrap" style={{flex: 1}}><AmbigVsIgnor /></div></div>
      </div>)},
  // 4b --------------------------------------------------------------- decomposition
  {title: "Measuring the set", steps: 1, notes: "How we turn the set into numbers: upper entropy (total), lower entropy (aleatoric), their difference (epistemic), and the interval width as an entropy-free size proxy. Keep the width in mind; it matters later.",
    render: () => (
      <div className="slide">
        <Header kicker="Background" title="Turning a set into numbers: upper and lower entropy, and width" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <div className="svgwrap" style={{flex: 1.3}}><DecompDiag /></div>
          <div className="col" style={{flex: 1, justifyContent: "center"}}>
            <Card tag="total" className="accent"><Eq small tex="\mathrm{TU}=\overline{H}=\max_{p\in\mathcal{C}}H(p)" /><p style={{textAlign: "center"}}>the most uncertain distribution the set still allows</p></Card>
            <Card tag="aleatoric" className="rose"><Eq small tex="\mathrm{AU}=\underline{H}=\min_{p\in\mathcal{C}}H(p)" /><p style={{textAlign: "center"}}>irreducible: even the most confident member sees this much ambiguity</p></Card>
            <Frag i={1}><Card tag="epistemic" style={{borderColor: "rgba(255,196,107,.45)"}}><Eq small tex="\mathrm{EU}=\overline{H}-\underline{H}\qquad\text{width}=\textstyle\sum_k(u_k-l_k)" /><p style={{textAlign: "center"}}>two candidate size measures; the study shows they behave very differently on unseen targets</p></Card></Frag>
          </div>
        </div></div>
      </div>)},
  // 5 ---------------------------------------------------------------- text?
  {title: "Credal on text?", steps: 0, notes: "Address the folk claim that credal networks only work on tabular data. True for classical credal networks; false for modern credal deep learning, where the set lives at the output.",
    render: () => (
      <div className="slide">
        <Header kicker="Background" title="“Credal networks only work on tabular data” — true for the classical family, not for ours" />
        <div className="body"><div className="svgwrap"><ClassicalVsModern /></div></div>
      </div>)},
  // 6 ---------------------------------------------------------------- design space
  {title: "Design space", steps: 6, notes: "Six ways to build the set on the same backbone. Walk through cost and what each one varies. Then the proposal fills the gap: none of them uses human disagreement as a training signal.",
    render: () => (
      <div className="slide">
        <Header kicker="Alternatives considered" title="Six credal constructions on one backbone" sub="Same DeBERTa-v3-base recipe everywhere, so only the credal construction differs." />
        <div className="body"><div className="grid3" style={{gridTemplateRows: "1fr 1fr", flex: 1}}>
          <Frag i={1} dim><Card tag="A1 · Wang 2025" title="CreWrap-DE"><p>Post-hoc wrapper over a deep ensemble: <b>l = min, u = max</b> over M seed/bootstrap members. Plug-and-play. Cost: M trainings.</p></Card></Frag>
          <Frag i={2} dim><Card tag="A2 · Gal 2016 + Wang 2025" title="MCD-Credal"><p>Same wrapper over T = 10 <b>MC-dropout</b> samples of one model. Cost: one training.</p></Card></Frag>
          <Frag i={3} dim><Card tag="A3 · Wang 2024" title="CreDE"><p><b>Interval heads</b> m ± h and an interval softmax give per-member [q_L, q_U]; trained with a DRO-style loss on the hardest half.</p></Card></Frag>
          <Frag i={4} dim><Card tag="A4 · 2026" title="CreDRO"><p>Members differ only by a <b>CVaR-DRO fraction δᵢ</b> ∈ [0.5, 1]: gradient from the hardest δᵢ of each batch. Disagreement by design.</p></Card></Frag>
          <Frag i={5} dim><Card tag="A5 · Hofman 2026" title="EffCre"><p><b>One model, no ensemble</b>: shift logits class-wise while the likelihood on a reference split stays within log α of its maximum; bisection finds the bounds.</p></Card></Frag>
          <Frag i={6} dim><Card tag="A6 · Manchingal 2026" title="Prompt-Credal (LLM)"><p>Zero-shot Qwen2.5-3B-Instruct, <b>8 prompt paraphrases</b>, restricted softmax over favor / against / none; min/max over prompts.</p></Card></Frag>
        </div></div>
      </div>)},
  // 7 ---------------------------------------------------------------- architecture
  {title: "Architecture", steps: 4, notes: "Build the pipeline left to right: encoder, members, credal layer, then the three consumers of the set.",
    render: (s) => (
      <div className="slide">
        <Header kicker="System architecture" title="From (text, target) to a calibrated, decision-ready credal set" />
        <div className="body"><div className="svgwrap"><Pipeline step={s} /></div></div>
      </div>)},
  // 8 ---------------------------------------------------------------- members
  {title: "Member objectives", steps: 0, notes: "The four training signals that produce member diversity. Every system in the study is one of these member types wrapped by the credal layer.",
    render: () => (
      <div className="slide">
        <Header kicker="Architecture · members" title="Four ways to train a member" sub="Same encoder, same hyper-parameters; only the loss changes." />
        <div className="body"><div className="grid4" style={{flex: 1}}>
          <Card tag="CE" title="Hard labels"><MemberMini kind="ce" /><Eq small tex="-\log p_y" /><p>standard cross-entropy; seed + bootstrap diversity only</p></Card>
          <Card tag="Soft" title="Human distribution" className="teal"><MemberMini kind="soft" /><Eq small tex="-\textstyle\sum_k \lambda_{x,k}\log p_k" /><p>λₓ from released per-annotator votes where they exist</p></Card>
          <Card tag="DRO" title="CVaR tail" style={{borderColor: "rgba(255,196,107,.45)"}}><MemberMini kind="dro" /><Eq small tex="\tfrac{1}{\lfloor\delta B\rfloor}\textstyle\sum_{i\in\mathrm{top}_\delta}\ell_i" /><p>member i uses δᵢ ∈ {"{0.5, 0.625, 0.75, 0.875, 1}"}</p></Card>
          <Card tag="CreNet" title="Interval head" style={{borderColor: "rgba(192,132,252,.5)"}}><MemberMini kind="crenet" /><Eq small tex="q_{L,i}=\sigma\!\big(a_{L,i}-\mathrm{LSE}_{k\ne i}\,m_k\big)" /><p>loss = CE(q_U) + CE(q_L) on the hardest half</p></Card>
        </div></div>
      </div>)},
  // 9 ---------------------------------------------------------------- proposal
  {title: "Proposed: CreDRO-Soft", steps: 3, notes: "The proposal. Two ingredients: soft labels from the released votes set the location; DRO member diversity sets the size. Evaluating each ingredient alone gives a 2x2 ablation.",
    render: (s) => (
      <div className="slide">
        <Header kicker="Proposed method" title={<>CreDRO-Soft: <span className="grad">location from people, size from robustness</span></>} />
        <div className="body"><div className="svgwrap"><CreDROSoftDiagram step={s} /></div></div>
      </div>)},
  // 10 --------------------------------------------------------------- why / 2x2
  {title: "Why it should work", steps: 2, notes: "State the two deficiencies of the alternatives and how the proposal addresses them. Then the 2x2.",
    render: () => (
      <div className="slide">
        <Header kicker="Proposed method" title="Why combine the two, and how we test that it matters" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <div className="col">
            <Step n="1" title="Hard labels teach nothing about contestedness">Every alternative trains on one-hot labels, so the lower entropy of its set is an artefact of the backbone, not a model of human disagreement. Soft labels pull every member towards λₓ, so <b>H̲ tracks human entropy</b>.</Step>
            <Step n="2" title="Seed diversity is incidental">Members disagree where they happen to, not where the data has shifted. DRO members with small δ concentrate on the hard tail, δ = 1 fits everything: they <b>diverge most on items unlike the training data</b>.</Step>
            <Step n="3" title="No new annotation">SemEval-2016 released ≈8 votes per tweet for four targets. We recover λₓ from that file; Climate Change keeps hard labels.</Step>
          </div>
          <Frag i={1} style={{flex: 1}}><Card title="The 2 × 2 ablation hidden in the method list" className="accent" style={{height: "100%"}}>
            <table className="t" style={{marginTop: 14, fontSize: 20}}>
              <thead><tr><th></th><th>seed diversity</th><th>DRO diversity</th></tr></thead>
              <tbody>
                <tr><td>hard labels</td><td>CreWrap-DE (A1)</td><td>CreDRO (A4)</td></tr>
                <tr><td>human λₓ</td><td>CreWrap-Soft</td><td className="good">CreDRO-Soft ★</td></tr>
              </tbody></table>
            <Frag i={2}><p style={{marginTop: 22}}>Each cell is a five-member ensemble built with identical settings, so a difference between cells is attributable to one ingredient.</p>
            <p>All four share the same credal layer, decomposition, conformal calibration and decision rule.</p></Frag>
          </Card></Frag>
        </div></div>
      </div>)},
  // 11 --------------------------------------------------------------- calibration + decision
  {title: "Calibration & decision", steps: 1, notes: "The two shared layers downstream of the set. Conformal dilation gives a coverage guarantee for the human distribution. Interval dominance gives the routing.",
    render: () => (
      <div className="slide">
        <Header kicker="Architecture · downstream layers" title="Guarantee the set, then act on it" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card tag="conformal credal calibration · Javanmardi 2024" style={{flex: 1}}><div className="svgwrap"><ConformalDiag /></div></Card>
          <Frag i={1} style={{flex: 1.6}}><Card tag="interval dominance · Troffaes 2007" style={{height: "100%"}}><div className="svgwrap"><Dominance /></div></Card></Frag>
        </div></div>
      </div>)},
  // 12 --------------------------------------------------------------- data & protocol
  {title: "Data & protocol", steps: 1, notes: "Splits and the leave-one-target-out design. Point out the calibration split is never trained on, and the 1,101 low-agreement tweets are a stress test the original task threw away.",
    render: () => (
      <div className="slide">
        <Header kicker="Experimental setup" title="Data, splits and the leave-one-target-out protocol" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <div style={{flex: 1.3}}>
            <table className="t">
              <thead><tr><th>Split</th><th>Items</th><th>With votes</th><th>Role</th></tr></thead>
              <tbody>
                <tr><td>SemEval-2016 train (80 %)</td><td>2,250</td><td>1,934</td><td>member training</td></tr>
                <tr><td>SemEval-2016 cal (20 %, stratified)</td><td>564</td><td>485</td><td>temperature · EffCre reference · conformal</td></tr>
                <tr><td>SemEval-2016 test (official)</td><td>1,249</td><td>1,080</td><td>in-target evaluation · human alignment</td></tr>
                <tr><td>Low-agreement pool (dropped from task)</td><td>1,385</td><td>1,385</td><td>ambiguity stress test (1,101 on seen targets)</td></tr>
                <tr><td>P-Stance test (Trump, Biden, Sanders)</td><td>2,157</td><td>–</td><td>cross-dataset unseen targets</td></tr>
                <tr><td>COVID-19-Stance test (4 targets)</td><td>800</td><td>–</td><td>cross-dataset unseen targets</td></tr>
              </tbody></table>
            <Frag i={1}><Card style={{marginTop: 22}}><p><b>Human label distributions</b> λₓ = (n_FAVOR, n_AGAINST, n_NONE) / n from the released SemEval-2016 annotator file, median n = 8; NEUTRAL votes mapped to NONE. Climate Change has no released votes and keeps hard labels.</p></Card></Frag>
          </div>
          <div className="col" style={{flex: 1}}>
            <Card tag="leave-one-target-out"><div className="svgwrap"><LOTO /></div></Card>
            <Frag i={1}><div className="grid2">
              <Card><p><b>Backbone</b> microsoft/deberta-v3-base, [CLS] pooling, AdamW 2e-5, 5 epochs, batch 32, max length 128, fp32 master weights.</p></Card>
              <Card><p><b>Scale</b> 166 member trainings (≈45 s each) on one RTX PRO 5000; all credal, conformal and decision computations on CPU from cached outputs.</p></Card>
            </div></Frag>
          </div>
        </div></div>
      </div>)},
  // 13 --------------------------------------------------------------- experiment map
  {title: "Experiment map", steps: 0, notes: "Eight experiments mapped to the research questions.",
    render: () => (
      <div className="slide">
        <Header kicker="Experimental setup" title="Eight experiments, four questions" />
        <div className="body"><div className="grid4" style={{gridTemplateRows: "1fr 1fr", flex: 1}}>
          <Card tag="E1 · RQ1" title="In-target quality"><p>accuracy, F_avg, macro-F1, ECE, NLL, misclassification AUROC, AURC, determinacy, set accuracy, u₆₅</p></Card>
          <Card tag="E4a · RQ2" title="Unseen targets (LOTO)"><p>AUROC seen vs unseen with EU, width and TU; unseen accuracy; 5 folds</p></Card>
          <Card tag="E4b · RQ2" title="Cross-dataset"><p>SemEval-trained models on P-Stance and COVID-19 test sets</p></Card>
          <Card tag="E5 · RQ2" title="Human alignment"><p>Spearman ρ of AU / EU / TU with human entropy; JSD and TV to λₓ</p></Card>
          <Card tag="E5b · RQ2" title="Ambiguity stress"><p>1,101 low-agreement tweets vs test: does EU stay flat while AU rises?</p></Card>
          <Card tag="E6 · RQ3" title="Conformal coverage"><p>coverage and efficiency over 100 random 50/50 re-splits of the voted items</p></Card>
          <Card tag="E7 · RQ4" title="Decision cost"><p>c_err = 1, c_defer = 0.3, c_set = 0.15 per extra class; confidence vs credal routing</p></Card>
          <Card tag="E8 + ablation" title="LLM & ensemble size"><p>prompt-credal with Qwen2.5-3B; M ∈ {"{1,2,3,4,5,8}"}; bootstrap CIs; paired fold tests</p></Card>
        </div></div>
      </div>)},
  // 14 --------------------------------------------------------------- E1
  {title: "E1 · Calibration", steps: 1, notes: "Calibration is where the credal structure pays off first. Accuracy is flat across methods; ECE drops by 3-5x. CIs: soft-label methods do not overlap with any hard-label method.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · E1 in-target (SemEval test, n = 1,249)" title="Same accuracy, three to five times better calibration" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1.3}}><HBar title="expected calibration error (15 bins) · lower is better" lowerBetter rows={rows(D.e1.filter(r => r[0] !== "LLM-PromptCredal"), 3)} max={0.17} /></Card>
          <div className="col" style={{flex: 1}}>
            <Card><table className="t"><thead><tr><th>Method</th><th>Acc</th><th>F_avg</th><th>ECE 95 % CI</th></tr></thead><tbody>
              {D.ci.map(r => <tr key={r[0]} className={r[0] === "CreDRO-Soft" ? "hl" : ""}><td>{r[0]}</td><td>{r[1].toFixed(3)}</td><td>{(D.e1.find(e => e[0] === r[0]) || [0, 0, 0])[2].toFixed(3)}</td><td className="mono">{r[2].toFixed(3)} [{r[3].toFixed(3)}, {r[4].toFixed(3)}]</td></tr>)}
            </tbody></table></Card>
            <Frag i={1}><Card className="rose"><p><b>Honest negative:</b> in-target <i>selective prediction</i> is not improved. Softmax entropy stays the best error detector (AUROC 0.789 vs 0.723), because credal total uncertainty is deliberately high on contested-but-correct items.</p></Card></Frag>
          </div>
        </div></div>
      </div>)},
  // 15 --------------------------------------------------------------- E4a
  {title: "E4a · Unseen targets", steps: 1, notes: "The headline. Upper entropy of CreDRO-Soft is the best and most stable novelty detector: 0.905 with std 0.021, wins all five folds against softmax, paired t-test p = 0.025. Note the fold variance: 3-4x smaller than baselines.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · E4a leave-one-target-out" title="Highest and most stable unseen-target detection" sub="Score = upper entropy of the set. Rings are the five folds; bar is ± one standard deviation." />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1.5}}><DotErr names={["Softmax", "TempScale", "DE-MI", "CreWrap-DE", "CreDE", "CreWrap-Soft", "CreDRO", "CreDRO-Soft", "EffCre"]} /></Card>
          <div className="col" style={{flex: 1}}>
            <Card className="accent"><div className="big grad">0.905<small>± 0.021</small></div><p style={{marginTop: 10}}>CreDRO-Soft vs softmax 0.816 ± 0.065 · <b>5/5 folds</b> · paired t p = 0.025 · Wilcoxon p = 0.063</p></Card>
            <Card><p><b>DRO diversity</b> alone: 0.897 ± 0.016 (CreDRO). <b>Soft labels</b> alone: 0.878 ± 0.034. Deep ensemble: 0.875 ± 0.037.</p><p>Fold variance of the credal DRO methods is 3–4× lower than every baseline.</p></Card>
            <Frag i={1}><Card className="teal"><p><b>Cross-dataset (E4b):</b> P-Stance AUROC 0.711 (CreWrap-Soft 0.713) vs softmax 0.597. On COVID every method is already high (0.85–0.90); EffCre leads at 0.904.</p></Card></Frag>
          </div>
        </div></div>
      </div>)},
  // 16 --------------------------------------------------------------- EU pathology
  {title: "EU pathology", steps: 0, notes: "A measurement result worth a slide of its own: the textbook EU = upper minus lower entropy fails on unseen targets in three-class stance. Members collapse to near-uniform, the lower entropy saturates, the difference shrinks. Use width or upper entropy.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · a measurement finding" title="The textbook epistemic score breaks on unseen targets" />
        <div className="body"><div className="svgwrap"><EUPathology /></div></div>
      </div>)},
  // 17 --------------------------------------------------------------- E5/E5b
  {title: "E5 · Ambiguity vs ignorance", steps: 1, notes: "The qualitative result the project set out to get. On the 1,101 contested tweets, the aleatoric score rises for every method, but only the soft-label credal models keep the epistemic score flat. Hard-label sets inflate EU on ambiguity, which is exactly the confusion we wanted to remove.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · E5 / E5b human disagreement" title="Only soft-label credal sets keep ignorance flat when ambiguity rises" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1.3}}><table className="t">
            <thead><tr><th>Method</th><th>ρ(AU, human H)</th><th>JSD(p*, λ)</th><th>mean AU test → ambig</th><th>mean EU test → ambig</th></tr></thead>
            <tbody>{D.e5.map(r => { const dEU = r[8] - r[7]; return (
              <tr key={r[0]} className={r[0] === "CreDRO-Soft" ? "hl" : ""}><td>{r[0]}</td><td className={r[1] >= 0.345 ? "good" : ""}>{r[1].toFixed(3)}</td><td className={r[3] <= 0.124 ? "good" : ""}>{r[3].toFixed(3)}</td>
                <td className="mono">{r[5].toFixed(2)} → {r[6].toFixed(2)}</td><td className={`mono ${Math.abs(dEU) < 0.03 ? "good" : dEU > 0.08 ? "bad" : ""}`}>{r[7].toFixed(2)} → {r[8].toFixed(2)} ({dEU >= 0 ? "+" : ""}{dEU.toFixed(2)})</td></tr>); })}
            </tbody></table>
            <p className="sub" style={{fontSize: 16, marginTop: 10}}>E5: n ≈ 1,080 test items with votes. E5b: 1,101 low-agreement tweets on the four seen targets vs the test set.</p></Card>
          <div className="col" style={{flex: 1}}>
            <Card className="imgcard" style={{flex: 1}}><img src="fig/au_eu_map.png" alt="AU/EU map" /></Card>
            <Frag i={1}><Card className="teal"><p><b>Reading the map:</b> ambiguity moves points <b>right</b> (AU up), novelty moves them <b>up</b> (width up). For CreDRO-Soft and CreWrap-Soft the two moves are orthogonal.</p></Card></Frag>
          </div>
        </div></div>
      </div>)},
  // 18 --------------------------------------------------------------- E6 + E7
  {title: "E6 · E7 coverage & cost", steps: 1, notes: "Coverage holds at the nominal level for everyone, as conformal theory promises; what differs is efficiency. Soft-label sets need the smallest dilation. Decision cost: credal routing at zero deferral beats confidence thresholding even at 20 percent deferral.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · E6 conformal coverage · E7 decision cost" title="Guaranteed coverage with tighter sets, and cheaper decisions without deferring" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1}}><HBar title="conformal quantile q̂ at 90 % coverage · smaller = tighter set" lowerBetter rows={rows(D.e6, 2)} max={0.8} /></Card>
          <Card style={{flex: 1}}><Grouped title="expected decision cost, in-target test · lower is better" labels={["confidence routing, 0 % deferral", "credal routing, 0 % deferral"]} rows={D.e7.map(r => ({name: r[0], v: [r[1], r[3]]}))} max={0.3} /></Card>
        </div>
          <Frag i={1}><div className="grid3" style={{marginTop: 18}}>
            <Card><p><b>Coverage</b> 0.90 at α = 0.1 and 0.80 at α = 0.2 for every method, as the theory predicts under exchangeability.</p></Card>
            <Card className="teal"><p><b>Efficiency</b> CreDRO-Soft needs q̂ = 0.542 vs 0.724 for softmax; its own box dilated by only 0.416 already covers 90 %.</p></Card>
            <Card className="accent"><p><b>Cost</b> credal routing cuts cost from 0.258 to 0.175–0.194 in-target, and 30–50 % on unseen targets, with no item deferred.</p></Card>
          </div></Frag>
        </div>
      </div>)},
  // 19 --------------------------------------------------------------- ablation
  {title: "Ablation · ensemble size", steps: 1, notes: "A single soft-label DRO model already beats a five-member hard-label ensemble on both calibration and novelty detection. Width-based scores need M >= 2 and saturate at 3. M = 8 brings nothing further.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · ablation" title="Most of the gain comes from the training signal, not the ensemble" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1}}><Lines metric="ece" yl={0.0} yh={0.16} title="in-target ECE vs ensemble size" fmt={v => v.toFixed(2)} /></Card>
          <Card style={{flex: 1}}><Lines metric="loto" yl={0.80} yh={0.94} title="LOTO unseen-target AUROC (upper entropy) vs ensemble size" fmt={v => v.toFixed(2)} /></Card>
        </div>
          <Frag i={1}><div className="grid3" style={{marginTop: 18}}>
            <Card className="accent"><p><b>M = 1 CreDRO-Soft:</b> ECE 0.066, AUROC 0.888. Already better than a five-member hard-label ensemble (0.066, 0.871) on both counts.</p></Card>
            <Card><p><b>M = 3</b> gives the best LOTO AUROC in the study (0.919). Width-based epistemic scores need M ≥ 2 and saturate at 3.</p></Card>
            <Card><p><b>M = 8</b> brings no further gain (ECE 0.045, COVID AUROC 0.881) while mean width keeps growing, so M = 5 is used throughout.</p></Card>
          </div></Frag>
        </div>
      </div>)},
  // 20 --------------------------------------------------------------- LLM
  {title: "E8 · LLM prompt-credal", steps: 0, notes: "Zero-shot LLM credal sets: calibrated by the prompt spread but far below the fine-tuned models in accuracy and in novelty detection. The 7B download stalled, so only the 3B is reported.",
    render: () => (
      <div className="slide">
        <Header kicker="Results · E8" title="Prompt-paraphrase credal sets from an LLM: cheaper, but not competitive" />
        <div className="body"><div className="row" style={{flex: 1}}>
          <Card style={{flex: 1.2}}><table className="t">
            <thead><tr><th>Qwen2.5-3B-Instruct, zero-shot</th><th>Acc</th><th>ECE</th><th>NLL</th><th>P-Stance AUROC</th><th>COVID AUROC</th><th>ρ(AU, H)</th></tr></thead>
            <tbody>
              <tr><td>LLM single prompt</td><td>0.635</td><td>0.274</td><td>2.063</td><td>0.415</td><td>0.430</td><td>0.092</td></tr>
              <tr><td>LLM prompt mixture (MI)</td><td>0.632</td><td>0.215</td><td>1.651</td><td>0.431</td><td>0.472</td><td>0.158</td></tr>
              <tr><td>LLM Prompt-Credal (8 paraphrases)</td><td>0.620</td><td className="good">0.143</td><td className="good">1.500</td><td>0.458</td><td>0.475</td><td>0.145</td></tr>
              <tr className="hl"><td>CreDRO-Soft (fine-tuned, 5 × 184M)</td><td>0.753</td><td>0.041</td><td>0.602</td><td>0.711</td><td>0.888</td><td>0.304</td></tr>
            </tbody></table></Card>
          <div className="col" style={{flex: 1}}>
            <Card><p><b>Credal over prompts does what it should:</b> ECE halves and NLL drops relative to a single prompt. Prompt paraphrase is a valid source of epistemic spread.</p></Card>
            <Card><p><b>But</b> zero-shot stance accuracy is 12 points lower and the LLM cannot tell P-Stance or COVID tweets from SemEval ones (AUROC below 0.5).</p></Card>
            <Card className="rose"><p>Qwen2.5-7B was attempted; the download stalled, so only the 3B results are reported. LoRA-ensemble credal LLMs remain future work.</p></Card>
          </div>
        </div></div>
      </div>)},
  // 21 --------------------------------------------------------------- contributions
  {title: "Contributions", steps: 5, notes: "Summarise: what the credal representation buys for stance, which construction, and the measurement finding.",
    render: () => (
      <div className="slide">
        <Header kicker="Discussion" title="What the credal representation buys for stance detection" />
        <div className="body"><div className="col" style={{gap: 20}}>
          <Frag i={1}><Step n="1" title="Calibrated probabilities without a post-hoc temperature">ECE 0.03–0.05 against 0.15 for softmax, at unchanged accuracy; confidence intervals do not overlap any hard-label method.</Step></Frag>
          <Frag i={2}><Step n="2" title="The most stable unseen-target detector in the study">Upper entropy of CreDRO-Soft: 0.905 ± 0.021 vs 0.816 ± 0.065, five folds out of five, p = 0.025.</Step></Frag>
          <Frag i={3}><Step n="3" title="Ambiguity and ignorance finally come apart">Soft-label credal sets are the only family whose epistemic score stays flat on 1,101 human-contested tweets while the aleatoric score rises.</Step></Frag>
          <Frag i={4}><Step n="4" title="Guaranteed, tighter, cheaper">Conformal coverage at the nominal level with the smallest dilation, and 25–50 % lower decision cost through set-valued answers with no deferral.</Step></Frag>
          <Frag i={5}><Step n="5" title="A methodological warning">EU = H̄ − H̲ is unusable for three-class novelty detection; use width or upper entropy. And a single soft-label DRO model is a strong cheap alternative when an ensemble is not affordable.</Step></Frag>
        </div></div>
      </div>)},
  // 22 --------------------------------------------------------------- limitations
  {title: "Limitations & future work", steps: 0, notes: "Be upfront about scope.",
    render: () => (
      <div className="slide">
        <Header kicker="Limitations and next steps" title="Where the evidence stops" />
        <div className="body"><div className="grid2" style={{flex: 1}}>
          <Card tag="limitations" title="Scope of the claims">
            <ul className="list">
              <li>One backbone size (DeBERTa-v3-base) and M = 5; no large backbone was tested.</li>
              <li>Human votes exist for four SemEval targets only; alignment claims are scoped to SemEval.</li>
              <li>Conformal guarantees assume exchangeability; cross-dataset sets are not guaranteed.</li>
              <li>In-target selective prediction is not improved over softmax entropy.</li>
              <li>Unseen-target accuracy is ≈0.5 for every method: this work improves <b>knowing</b> when the model is out of its depth, not the stance itself.</li>
            </ul></Card>
          <Card tag="future work" title="Natural extensions" className="teal">
            <ul className="list">
              <li>ChaosNLI-style data with ≈100 votes per item to tighten the 90 % guarantee.</li>
              <li>Distance-based epistemic measures (Gonzalez-Garcia et al. 2026) in place of width.</li>
              <li>LoRA-ensemble credal LLMs (Manchingal et al. 2026) rather than zero-shot prompting.</li>
              <li>Target-adaptive conformal calibration for cross-dataset deployment.</li>
              <li>Human-in-the-loop study of the predict / set / defer routing with real analysts.</li>
            </ul></Card>
        </div></div>
      </div>)},
  // 23 --------------------------------------------------------------- references
  {title: "References", steps: 0, notes: "",
    render: () => (
      <div className="slide">
        <Header kicker="References" title="Key sources" />
        <div className="body twocol-refs" style={{display: "block"}}>
          {[
            "Abellán, Klir, Moral (2006). Disaggregated total uncertainty measure for credal sets. <i>Int. J. General Systems</i>.",
            "Corani, Zaffalon (2008). The naive credal classifier 2. <i>JMLR</i>.",
            "Cozman (2000). Credal networks. <i>Artificial Intelligence</i>.",
            "Gal, Ghahramani (2016). Dropout as a Bayesian approximation. <i>ICML</i>.",
            "Glandt, Khanal, Li, Caragea, Caragea (2021). Stance detection in COVID-19 tweets. <i>ACL</i>.",
            "Gonzalez-Garcia et al. (2026). Quantification of credal uncertainty with distance-based measures. <i>arXiv</i>.",
            "Guo, Pleiss, Sun, Weinberger (2017). On calibration of modern neural networks. <i>ICML</i>.",
            "He, Gao, Chen (2023). DeBERTaV3. <i>ICLR</i>.",
            "Hofman, Löhr, Muschalik, Sale, Hüllermeier (2026). Efficient credal prediction through decalibration. <i>ICLR</i>.",
            "Hüllermeier, Waegeman (2021). Aleatoric and epistemic uncertainty in machine learning. <i>Machine Learning</i>.",
            "Javanmardi, Stutz, Hüllermeier (2024). Conformalized credal set predictors. <i>NeurIPS</i>.",
            "Lakshminarayanan, Pritzel, Blundell (2017). Deep ensembles. <i>NeurIPS</i>.",
            "Learning credal ensembles via distributionally robust optimization (CreDRO) (2026). <i>arXiv:2602.08470</i>.",
            "Li et al. (2021). P-Stance: A large dataset for stance detection in political domain. <i>Findings of ACL</i>.",
            "Löhr, Hofman, Mohr, Hüllermeier (2025). Credal prediction based on relative likelihood. <i>NeurIPS</i>.",
            "Manchingal, Nikolenko, Cuzzolin (2026). Credal large language models. <i>arXiv:2608.23244</i>.",
            "Mohammad, Kiritchenko, Sobhani, Zhu, Cherry (2016). SemEval-2016 Task 6: Detecting stance in tweets. <i>SemEval</i>.",
            "Mohammad, Sobhani, Kiritchenko (2017). Stance and sentiment in tweets. <i>ACM TOIT</i>. Raw annotations released with the paper.",
            "Nie, Zhou, Bansal (2020). ChaosNLI. <i>EMNLP</i>.",
            "Sale, Caprio, Hüllermeier (2023). Is the volume of a credal set a good measure for epistemic uncertainty? <i>UAI</i>.",
            "Sensoy, Kaplan, Kandemir (2018). Evidential deep learning. <i>NeurIPS</i>.",
            "Troffaes (2007). Decision making under uncertainty using imprecise probabilities. <i>IJAR</i>.",
            "Walley (1991). <i>Statistical Reasoning with Imprecise Probabilities</i>. Chapman & Hall.",
            "Wang, Cuzzolin, Manchingal, Shariatmadar, Moens, Hallez (2024). Credal deep ensembles. <i>NeurIPS</i>.",
            "Wang, Cuzzolin et al. (2025). Credal wrapper of model averaging. <i>ICLR</i>.",
            "Wimmer, Sale, Hofman, Bischl, Hüllermeier (2023). Quantifying aleatoric and epistemic uncertainty: are conditional entropy and mutual information appropriate measures? <i>UAI</i>.",
            "Zaffalon, Corani, Mauá (2012). Evaluating credal classifiers by utility-discounted predictive accuracy. <i>IJAR</i>.",
            "Zhao, Caragea (2024). EZ-STANCE. <i>ACL</i>.",
          ].map((r, i) => <p key={i} className="ref" dangerouslySetInnerHTML={{__html: r}} />)}
        </div>
      </div>)},
  // 24 --------------------------------------------------------------- end
  {title: "Thank you", steps: 0, notes: "",
    render: () => (
      <div className="slide title-slide">
        <div className="kicker">Thank you</div>
        <h1 style={{fontSize: 68}}>Questions?</h1>
        <p className="sub" style={{fontSize: 24, marginTop: 20}}>Code, data, cached model outputs, the full report and this presentation are in one repository.</p>
        <div className="title-meta">
          <Pill kind="acc">github.com/RakeshTirlangi/Credi-Stance</Pill><Pill kind="teal">FINAL_REPORT.md</Pill><Pill kind="gold">IMPLEMENTATION.md</Pill>
        </div>
        <div style={{display: "flex", gap: 60, marginTop: 60}}>
          <div className="hero-num"><div className="v grad">6 + 1</div><div className="k">credal constructions compared, one proposed</div></div>
          <div className="hero-num"><div className="v grad">8</div><div className="k">experiments, 5 LOTO folds, 100 conformal re-splits</div></div>
          <div className="hero-num"><div className="v grad">0</div><div className="k">new annotations collected</div></div>
        </div>
      </div>)},
];
