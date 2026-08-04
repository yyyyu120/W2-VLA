"use client";

/* eslint-disable @next/next/no-img-element -- paper figures are pre-rendered, responsive static assets */

import { useMemo, useState } from "react";
import type { CSSProperties } from "react";

type NumericCell = number | null;

const liberoRows: Array<{
  method: string;
  scores: [number, number, number, number, number];
  accent?: boolean;
}> = [
  { method: "OpenVLA", scores: [84.7, 88.4, 79.2, 53.7, 76.5] },
  { method: "OpenVLA-OFT", scores: [97.6, 98.4, 97.9, 94.5, 97.1] },
  { method: "π₀", scores: [96.8, 98.8, 95.8, 85.2, 94.2] },
  { method: "π₀-Fast", scores: [96.4, 96.8, 88.6, 60.2, 85.5] },
  { method: "π₀.₅", scores: [98.8, 98.2, 98.0, 92.4, 96.9] },
  { method: "Fast-ThinkAct", scores: [92.0, 97.2, 90.2, 79.4, 89.7] },
  { method: "DreamVLA", scores: [97.5, 94.0, 89.5, 89.5, 92.6] },
  { method: "GR00T-N1", scores: [94.4, 97.6, 93.0, 90.6, 93.9] },
  { method: "MemoryVLA", scores: [98.4, 98.4, 96.4, 93.4, 96.7] },
  { method: "VLA-JEPA", scores: [96.2, 99.6, 97.2, 95.8, 97.2] },
  { method: "DeepThinkVLA", scores: [96.6, 99.0, 96.4, 96.2, 97.0] },
  { method: "StarVLA", scores: [97.8, 98.8, 97.4, 92.0, 96.5] },
  { method: "W²-VLA", scores: [99.6, 99.8, 99.2, 95.2, 98.5], accent: true },
];

const robotwinRows: Array<{
  method: string;
  easy: NumericCell;
  hard: NumericCell;
  accent?: boolean;
}> = [
  { method: "π₀", easy: 46.42, hard: 16.34 },
  { method: "RDT", easy: 34.5, hard: 13.72 },
  { method: "Diffusion Policy", easy: 28.06, hard: 0.64 },
  { method: "UP-VLA", easy: 52.92, hard: 15.16 },
  { method: "StarVLA-OFT", easy: 50.38, hard: null },
  { method: "StarVLA-GR00T", easy: 48.8, hard: null },
  { method: "W²-VLA", easy: 60.71, hard: 18.21, accent: true },
];

const ablationPanels = {
  components: {
    label: "Components",
    headers: ["Configuration", "Spatial", "Object", "Goal", "Long", "Avg."],
    rows: [
      ["W²-VLA", "99.6", "99.8", "99.2", "95.2", "98.5"],
      ["w/o Wrist Predictor", "98.6", "99.6", "98.2", "93.6", "97.5"],
      ["w/o W²-CoT", "99.0", "99.2", "98.8", "95.0", "98.0"],
    ],
    note: "Future-wrist prediction contributes the largest gain on the temporally extended Long suite.",
  },
  interface: {
    label: "Interface & latency",
    headers: ["Configuration", "Decode CoT", "Wrist predictor", "Latent tokens", "Latency", "Avg."],
    rows: [
      ["1", "Yes", "No", "N/A", "1550.77 ms", "97.6"],
      ["2", "Yes", "Yes", "N/A", "1615.27 ms", "98.1"],
      ["3", "No", "Yes", "4", "98.58 ms", "98.0"],
      ["4", "No", "Yes", "8", "102.15 ms", "98.1"],
      ["5", "No", "Yes", "32", "148.69 ms", "98.4"],
      ["W²-VLA", "No", "Yes", "16", "110.58 ms", "98.5"],
    ],
    note: "The 16-token latent interface preserves the best average score while avoiding autoregressive CoT latency.",
  },
  targets: {
    label: "Prediction targets",
    headers: ["Configuration", "Main-view", "Wrist-view", "Latency", "Avg."],
    rows: [
      ["1", "Yes", "No", "102.49 ms", "97.7"],
      ["2", "Yes", "Yes", "132.76 ms", "98.0"],
      ["W²-VLA", "No", "Yes", "110.58 ms", "98.5"],
    ],
    note: "Wrist-only prediction focuses the objective on action-proximal contact, alignment, and release dynamics.",
  },
} as const;

const realWorld = {
  Standard: {
    success: [
      [36.67, 60, 26.67],
      [53.33, 80, 30],
      [63.33, 86.67, 60],
    ],
    progress: [
      [2.06, 2.33, 1.93],
      [2.96, 2.63, 1.86],
      [3.33, 2.77, 2.6],
    ],
  },
  OOD: {
    success: [
      [26.67, 46.67, 3.33],
      [36.67, 66.67, 10],
      [50, 73.33, 33.33],
    ],
    progress: [
      [1.8, 2.03, 1.43],
      [2.53, 2.43, 1.53],
      [2.83, 2.5, 2.27],
    ],
  },
} as const;

const methodNames = ["π₀", "VLA-JEPA", "W²-VLA"];
const taskNames = ["Table Cleaning", "Occluded Placement", "Bimanual Plug Insertion"];
const taskMax = [4, 3, 3];

const realWorldVideos = [
  {
    label: "Table Cleaning",
    folder: "task1_table_cleaning",
  },
  {
    label: "Occluded Placement",
    folder: "task2_occluded_placement",
  },
  {
    label: "Bimanual Plug Insertion",
    folder: "task3_bimanual_plug_insertion",
  },
] as const;

const videoVariants = [
  { label: "Normal", file: "normal_demo.mp4" },
  { label: "Background variation", file: "background_variation.mp4" },
  { label: "Light variation", file: "light_variation.mp4" },
  { label: "Clutter table", file: "clutter_table.mp4" },
] as const;

function SectionHeading({
  eyebrow,
  title,
  copy,
}: {
  eyebrow: string;
  title: string;
  copy?: string;
}) {
  return (
    <div className="section-heading">
      <span className="eyebrow">{eyebrow}</span>
      <div className="section-title-row">
        <h2>{title}</h2>
        {copy ? <p>{copy}</p> : null}
      </div>
    </div>
  );
}

function FigureCard({
  src,
  alt,
  caption,
  className = "",
}: {
  src: string;
  alt: string;
  caption: string;
  className?: string;
}) {
  return (
    <figure className={`figure-card ${className}`}>
      <div className="figure-media">
        <img src={src} alt={alt} loading="lazy" />
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}

function TeaserVideo() {
  return (
    <figure className="figure-card hero-figure teaser-video-card">
      <div className="figure-media teaser-video-frame">
        <video
          autoPlay
          controls
          loop
          muted
          playsInline
          poster="/assets/figures/teaser.png"
          preload="auto"
          aria-label="World-to-Wrist VLA project teaser video"
        >
          <source
            src="https://huggingface.co/datasets/HarrisonPENG/W2-VLA-assets/resolve/main/world-to-wrist-mute-16x9.mp4"
            type="video/mp4"
          />
          Your browser does not support embedded video. You can watch it directly on Hugging Face.
        </video>
      </div>
    </figure>
  );
}

function RealWorldVideoExplorer() {
  const [taskIndex, setTaskIndex] = useState(0);
  const [variantIndex, setVariantIndex] = useState(0);
  const task = realWorldVideos[taskIndex];
  const variant = videoVariants[variantIndex];
  const source = `https://huggingface.co/datasets/HarrisonPENG/W2-VLA-assets/resolve/main/${task.folder}/${variant.file}`;
  const folderUrl = `https://huggingface.co/datasets/HarrisonPENG/W2-VLA-assets/tree/main/${task.folder}`;

  return (
    <div className="demo-video-shell">
      <div className="demo-task-tabs" role="tablist" aria-label="Real-world video tasks">
        {realWorldVideos.map((item, index) => (
          <button
            key={item.folder}
            type="button"
            role="tab"
            aria-selected={taskIndex === index}
            className={taskIndex === index ? "active" : ""}
            onClick={() => setTaskIndex(index)}
          >
            <span>0{index + 1}</span>
            {item.label}
          </button>
        ))}
      </div>

      <div className="demo-video-stage">
        <video
          key={source}
          autoPlay
          controls
          loop
          muted
          playsInline
          preload="metadata"
          aria-label={`${task.label}, ${variant.label} demonstration`}
        >
          <source src={source} type="video/mp4" />
          Your browser does not support embedded video.
        </video>
      </div>

      <div className="demo-video-toolbar">
        <div className="demo-variant-tabs" aria-label="Video conditions">
          {videoVariants.map((item, index) => (
            <button
              key={item.file}
              type="button"
              aria-pressed={variantIndex === index}
              className={variantIndex === index ? "active" : ""}
              onClick={() => setVariantIndex(index)}
            >
              {item.label}
            </button>
          ))}
        </div>
        <a href={folderUrl} target="_blank" rel="noreferrer">View source videos ↗</a>
      </div>
    </div>
  );
}

function BenchmarkExplorer() {
  const [tab, setTab] = useState<"libero" | "robotwin" | "ablation">("libero");
  const [ablation, setAblation] = useState<keyof typeof ablationPanels>("components");

  return (
    <div className="results-shell">
      <div className="tab-list" role="tablist" aria-label="Benchmark result tables">
        {[
          ["libero", "LIBERO"],
          ["robotwin", "RoboTwin 2.0"],
          ["ablation", "Ablations"],
        ].map(([key, label]) => (
          <button
            key={key}
            type="button"
            role="tab"
            aria-selected={tab === key}
            className={tab === key ? "active" : ""}
            onClick={() => setTab(key as typeof tab)}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "libero" ? (
        <div className="result-panel" role="tabpanel">
          <div className="panel-intro">
            <div>
              <span className="panel-kicker">40 tasks · 2,000 evaluation trials</span>
              <h3>LIBERO suite success rate</h3>
              <p>W²-VLA leads the aggregate and three of four suites, reaching a 98.5% overall average.</p>
            </div>
            <div className="big-number"><strong>+1.3</strong><span>pts over the strongest baseline</span></div>
          </div>
          <div className="split-results">
            <div className="rank-chart" aria-label="LIBERO average score comparison">
              {[...liberoRows]
                .sort((a, b) => b.scores[4] - a.scores[4])
                .map((row, index) => (
                  <div className={`rank-row ${row.accent ? "accent" : ""}`} key={row.method}>
                    <span className="rank-index">{String(index + 1).padStart(2, "0")}</span>
                    <span className="rank-name">{row.method}</span>
                    <span className="rank-track">
                      <span style={{ width: `${row.scores[4]}%` }} />
                    </span>
                    <strong>{row.scores[4].toFixed(1)}</strong>
                  </div>
                ))}
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr><th scope="col">Method</th><th scope="col">Spatial</th><th scope="col">Object</th><th scope="col">Goal</th><th scope="col">Long</th><th scope="col">Avg.</th></tr>
                </thead>
                <tbody>
                  {liberoRows.map((row) => (
                    <tr className={row.accent ? "highlight" : ""} key={row.method}>
                      <th scope="row">{row.method}</th>
                      {row.scores.map((score, i) => <td key={i}>{score.toFixed(1)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      ) : null}

      {tab === "robotwin" ? (
        <div className="result-panel" role="tabpanel">
          <div className="panel-intro">
            <div>
              <span className="panel-kicker">50 bimanual tasks · clean and domain-randomized</span>
              <h3>RoboTwin 2.0 success rate</h3>
              <p>The gain persists under both Easy and Hard settings, demonstrating stronger bimanual generalization.</p>
            </div>
            <div className="big-number"><strong>60.71</strong><span>Easy setting success</span></div>
          </div>
          <div className="robotwin-chart" aria-label="RoboTwin easy and hard score comparison">
            <div className="chart-legend"><span className="legend-easy">Easy</span><span className="legend-hard">Hard</span></div>
            {robotwinRows.map((row) => (
              <div className={`twin-row ${row.accent ? "accent" : ""}`} key={row.method}>
                <span>{row.method}</span>
                <div className="twin-bars">
                  <i className="easy" style={{ width: `${(row.easy ?? 0) / 0.65}%` }}><b>{row.easy?.toFixed(2) ?? "—"}</b></i>
                  <i className="hard" style={{ width: `${(row.hard ?? 0) / 0.2}%` }}><b>{row.hard?.toFixed(2) ?? "—"}</b></i>
                </div>
              </div>
            ))}
          </div>
          <div className="table-wrap compact-table">
            <table>
              <thead><tr><th scope="col">Method</th><th scope="col">Easy</th><th scope="col">Hard</th></tr></thead>
              <tbody>{robotwinRows.map((row) => (
                <tr className={row.accent ? "highlight" : ""} key={row.method}>
                  <th scope="row">{row.method}</th><td>{row.easy?.toFixed(2) ?? "—"}</td><td>{row.hard?.toFixed(2) ?? "—"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      ) : null}

      {tab === "ablation" ? (
        <div className="result-panel" role="tabpanel">
          <div className="panel-intro">
            <div>
              <span className="panel-kicker">LIBERO ablation study</span>
              <h3>What makes World-to-Wrist work?</h3>
              <p>Inspect component contributions, the latent interface, and the future-view prediction target.</p>
            </div>
          </div>
          <div className="subtabs" aria-label="Ablation panels">
            {(Object.keys(ablationPanels) as Array<keyof typeof ablationPanels>).map((key) => (
              <button key={key} type="button" className={ablation === key ? "active" : ""} onClick={() => setAblation(key)}>{ablationPanels[key].label}</button>
            ))}
          </div>
          <div className="ablation-callout"><span>Key readout</span><p>{ablationPanels[ablation].note}</p></div>
          <div className="table-wrap">
            <table>
              <thead><tr>{ablationPanels[ablation].headers.map((header) => <th scope="col" key={header}>{header}</th>)}</tr></thead>
              <tbody>{ablationPanels[ablation].rows.map((row, index) => (
                <tr className={row[0] === "W²-VLA" ? "highlight" : ""} key={`${row[0]}-${index}`}>
                  {row.map((cell, cellIndex) => cellIndex === 0 ? <th scope="row" key={cellIndex}>{cell}</th> : <td key={cellIndex}>{cell}</td>)}
                </tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      ) : null}

    </div>
  );
}

function RealWorldExplorer() {
  const [condition, setCondition] = useState<keyof typeof realWorld>("Standard");
  const [metric, setMetric] = useState<"success" | "progress">("success");
  const values = realWorld[condition][metric];
  const summary = useMemo(() => values.map((method) => method.reduce((sum, value) => sum + value, 0) / method.length), [values]);

  return (
    <div className="real-world-shell">
      <div className="real-controls">
        <div>
          <span>Condition</span>
          <div className="segmented">
            {(["Standard", "OOD"] as const).map((item) => <button type="button" key={item} aria-pressed={condition === item} className={condition === item ? "active" : ""} onClick={() => setCondition(item)}>{item}</button>)}
          </div>
        </div>
        <div>
          <span>Metric</span>
          <div className="segmented">
            <button type="button" aria-pressed={metric === "success"} className={metric === "success" ? "active" : ""} onClick={() => setMetric("success")}>Success rate</button>
            <button type="button" aria-pressed={metric === "progress"} className={metric === "progress" ? "active" : ""} onClick={() => setMetric("progress")}>Progress score</button>
          </div>
        </div>
      </div>

      <div className="real-content">
        <div className="vertical-chart" aria-label={`${condition} ${metric} comparison`}>
          <div className="chart-topline"><span>{metric === "success" ? "Success rate (%)" : "Ordered progress score"}</span><span>higher is better ↑</span></div>
          <div className="chart-area">
            {taskNames.map((task, taskIndex) => (
              <div className="bar-cluster" key={task}>
                <div className="bars">
                  {methodNames.map((method, methodIndex) => {
                    const value = values[methodIndex][taskIndex];
                    const normalized = metric === "success" ? value : (value / taskMax[taskIndex]) * 100;
                    return (
                      <div className="bar-slot" key={method}>
                        <span className={`bar method-${methodIndex}`} style={{ "--bar-height": `${normalized}%` } as CSSProperties}><b>{value.toFixed(metric === "success" ? 1 : 2)}</b></span>
                      </div>
                    );
                  })}
                </div>
                <strong>{task}</strong>
                <small>{metric === "progress" ? `max ${taskMax[taskIndex]}` : "30 trials / method"}</small>
              </div>
            ))}
          </div>
          <div className="method-legend">{methodNames.map((method, index) => <span className={`method-${index}`} key={method}>{method}</span>)}</div>
        </div>

        <aside className="real-summary">
          <span className="panel-kicker">Average across tasks</span>
          <h3>{condition} · {metric === "success" ? "success" : "progress"}</h3>
          <div className="summary-list">
            {methodNames.map((method, index) => (
              <div className={index === 2 ? "winner" : ""} key={method}><span>{method}</span><strong>{summary[index].toFixed(metric === "success" ? 2 : 2)}{metric === "success" ? "%" : ""}</strong></div>
            ))}
          </div>
          <p>W²-VLA ranks first on every task in both Standard and OOD conditions.</p>
        </aside>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main id="top">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="W squared VLA home">
          <span className="brand-mark">W²</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#overview">Overview</a>
          <a href="#method">Method</a>
          <a href="#results">Benchmarks</a>
          <a href="#real-world">Real world</a>
          <a href="#visuals">Visuals</a>
        </nav>
        <a className="header-cta" href="#results">Explore results <span>↘</span></a>
      </header>

      <section className="hero" id="overview">
        <div className="hero-glow hero-glow-one" />
        <div className="hero-glow hero-glow-two" />
        <div className="hero-copy">
          <h1>
            <em>World-to-Wrist</em>
            <span>Task-Conditioned Future Wrist Modeling</span>
            <span>for Fine-Grained Robot Manipulation</span>
          </h1>

          <div className="author-block" aria-label="Authors and affiliations">
            <div className="author-row">
              <span>Yuhao Pan<sup>1,*</sup></span>
              <span>Haosong Peng<sup>1,*</sup></span>
              <span>Zhengshen Zhang<sup>2</sup></span>
              <span>Zhengyang Yan<sup>1</sup></span>
            </div>
            <div className="author-row">
              <span>Yalun Dai<sup>3</sup></span>
              <span>Fushuo Huo<sup>7</sup></span>
              <span>Chujie Wang<sup>4</sup></span>
              <span>Tianyu Qi<sup>5</sup></span>
            </div>
            <div className="author-row">
              <span>Xiucheng Wang<sup>6</sup></span>
              <span>Nan Cheng<sup>6</sup></span>
              <span>Wenchao Xu<sup>1,†</sup></span>
            </div>
            <div className="affiliations">
              <p><sup>1</sup>The Hong Kong University of Science and Technology <span>·</span> <sup>2</sup>National University of Singapore</p>
              <p><sup>3</sup>Nanyang Technological University <span>·</span> <sup>4</sup>Wuhan University <span>·</span> <sup>5</sup>Sun Yat-sen University</p>
              <p><sup>6</sup>Xidian University <span>·</span> <sup>7</sup>Southeast University</p>
              <p className="author-notes"><sup>*</sup>Equal contribution. <span>·</span> <sup>†</sup>Corresponding author.</p>
            </div>
          </div>

          <div className="resource-actions" aria-label="Project links">
            <a className="resource-button primary" href="https://huggingface.co/papers/2605.27367" target="_blank" rel="noreferrer"><span className="button-icon">P</span>Paper</a>
            <a className="resource-button" href="https://arxiv.org/abs/2605.27367" target="_blank" rel="noreferrer"><span className="button-icon">A</span>arXiv</a>
            <a className="resource-button" href="https://ropedia.github.io/SpatialBench" target="_blank" rel="noreferrer"><span className="button-icon">P</span>Project Page</a>
            <a className="resource-button" href="https://huggingface.co/ropedia-ai/DA-Next" target="_blank" rel="noreferrer"><span className="button-icon">M</span>W²-VLA</a>
            <a className="resource-button" href="https://huggingface.co/datasets/yuuu94/W2-VLA-CoT" target="_blank" rel="noreferrer"><span className="button-icon">C</span>W²-CoT</a>
            <a className="resource-button" href="https://huggingface.co/datasets/yuuu94/W2-VLA-Training-Data" target="_blank" rel="noreferrer"><span className="button-icon">D</span>Dataset</a>
          </div>

          <p>W²-VLA turns global task understanding into action-proximal foresight by predicting future wrist latents from a compact, task-conditioned interface and wrist history.</p>
        </div>

        <div className="hero-stats" aria-label="Key results">
          <div><strong>98.5<small>%</small></strong><span>LIBERO average</span></div>
          <div><strong>60.71<small>%</small></strong><span>RoboTwin Easy</span></div>
          <div><strong>18.21<small>%</small></strong><span>RoboTwin Hard</span></div>
          <div><strong>80<small>+ Hz</small></strong><span>real-time generation</span></div>
        </div>

        <TeaserVideo />
      </section>

      <section className="abstract-section page-section">
        <h2 className="abstract-title">Abstract</h2>
        <div className="abstract-box">
          <p>Vision-language-action (VLA) models often treat main-view and wrist-view observations as parallel visual inputs, overlooking their distinct roles in robot manipulation. Fine-grained manipulation, however, benefits from anticipating how wrist-local interactions may evolve under the global task context. To address this limitation, we present <strong>World-to-Wrist VLA (W²-VLA)</strong>, a VLA model for fine-grained robot manipulation with task-conditioned future wrist modeling. Given current multi-view observations and an instruction, W²-VLA contextualizes a set of latent modeling tokens as a compact interface between the VLM and the wrist predictor. Conditioned on this interface and wrist history, the wrist predictor forecasts future wrist latents, which are converted into future-aware context for action prediction. In addition, we propose <strong>W²-CoT</strong>, a synthesis pipeline that produces structured annotations for manipulation progress, physical transition cues, and wrist-local evidence. These structured annotations provide auxiliary supervision to help shape the task-conditioned latent interface. Experiments on LIBERO, RoboTwin 2.0, and real-world manipulation tasks demonstrate improved fine-grained and contact-sensitive manipulation across single-arm and bimanual settings, while maintaining real-time action-generation above 80 Hz.</p>
        </div>
        <div className="contribution-strip">
          <article><span>01</span><h3>Directional modeling</h3><p>Global world context is explicitly routed toward wrist-local dynamics.</p></article>
          <article><span>02</span><h3>Latent foresight</h3><p>Future wrist representations capture contact, alignment, and release.</p></article>
          <article><span>03</span><h3>Fast inference</h3><p>A fixed-length latent interface avoids explicit CoT generation at deployment.</p></article>
        </div>
      </section>

      <section className="method-section page-section" id="method">
        <SectionHeading eyebrow="Method" title="A two-branch policy joined by one task-conditioned interface." />
        <FigureCard
          className="method-figure"
          src="/assets/figures/overview.png"
          alt="Architecture of W squared VLA with a Qwen VLM world branch, V-JEPA wrist branch, predictor, adapter, and DiT action head"
          caption="W²-VLA overview. Future wrist clips are training-only targets; deployed inference uses current observations, wrist history, and the instruction."
        />
        <div className="method-cards">
          <article><div className="node blue-node">World</div><span>①</span><h3>Task-conditioned interface</h3><p>Current multi-view observations and language contextualize 16 latent modeling tokens inside Qwen-VL.</p></article>
          <article><div className="node green-node">Wrist</div><span>②</span><h3>Future wrist predictor</h3><p>Frozen V-JEPA wrist-history features are queried by task-conditioned states to forecast local future latents.</p></article>
          <article><div className="node orange-node">Action</div><span>③</span><h3>Future-aware control</h3><p>A lightweight adapter fuses the predicted wrist context with VLM states for flow-matching action generation.</p></article>
        </div>
      </section>

      <section className="results-section page-section" id="results">
        <SectionHeading eyebrow="Benchmark results" title="Strong across single-arm and bimanual manipulation." />
        <BenchmarkExplorer />
      </section>

      <section className="real-section page-section" id="real-world">
        <SectionHeading eyebrow="Real-world evaluation" title="Robust progress, even when the world changes." />
        <RealWorldExplorer />
      </section>

      <section className="visual-section page-section" id="visuals">
        <SectionHeading eyebrow="Paper visuals" title="From task-level plans to wrist-local evidence." />
        <div className="visual-grid">
          <FigureCard
            className="wide"
            src="/assets/figures/rollouts.png"
            alt="Real-world rollouts for table cleaning, occluded placement, and bimanual plug insertion"
            caption="Real-world rollouts span long-horizon cleaning, obstacle-aware placement, and contact-rich bimanual insertion."
          />
          <FigureCard
            className="wide"
            src="/assets/figures/visual.png"
            alt="Attention visualizations over main and wrist views during simulation and real-world manipulation"
            caption="Latent modeling token attention tracks task-relevant evidence across main and wrist views. CoT text is decoded for visualization only."
          />
        </div>
      </section>

      <section className="video-section page-section" id="videos">
        <SectionHeading eyebrow="Real-world videos" title="Three tasks across four visual conditions." />
        <RealWorldVideoExplorer />
      </section>

      <section className="citation-section" id="citation">
        <div className="citation-heading">
          <span className="eyebrow">Citation</span>
          <h2>BibTeX</h2>
        </div>
        <pre aria-label="BibTeX citation placeholder"><code>{`@article{w2vla_placeholder,
  title   = {World-to-Wrist: Task-Conditioned Future Wrist Modeling for Fine-Grained Robot Manipulation},
  author  = {Author list to be updated},
  journal = {Venue to be updated},
  year    = {2027}
}`}</code></pre>
      </section>

      <footer><div className="brand footer-brand"><span className="brand-mark">W²</span></div><p>World-to-Wrist: Task-Conditioned Future Wrist Modeling for Fine-Grained Robot Manipulation.</p></footer>
    </main>
  );
}
