"use strict";

const liberoRows = [
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

const robotwinRows = [
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
};

const realWorld = {
  Standard: {
    success: [[36.67, 60, 26.67], [53.33, 80, 30], [63.33, 86.67, 60]],
    progress: [[2.06, 2.33, 1.93], [2.96, 2.63, 1.86], [3.33, 2.77, 2.6]],
  },
  OOD: {
    success: [[26.67, 46.67, 3.33], [36.67, 66.67, 10], [50, 73.33, 33.33]],
    progress: [[1.8, 2.03, 1.43], [2.53, 2.43, 1.53], [2.83, 2.5, 2.27]],
  },
};

const methodNames = ["π₀", "VLA-JEPA", "W²-VLA"];
const taskNames = ["Table Cleaning", "Occluded Placement", "Bimanual Plug Insertion"];
const taskMax = [4, 3, 3];
const videoTasks = [
  { label: "Table Cleaning", folder: "task1_table_cleaning" },
  { label: "Occluded Placement", folder: "task2_occluded_placement" },
  { label: "Bimanual Plug Insertion", folder: "task3_bimanual_plug_insertion" },
];
const videoVariants = [
  { label: "Normal", file: "normal_demo.mp4" },
  { label: "Background variation", file: "background_variation.mp4" },
  { label: "Light variation", file: "light_variation.mp4" },
  { label: "Clutter table", file: "clutter_table.mp4" },
];

const benchmarkPanel = document.querySelector("#benchmark-panel");
let activeBenchmark = "libero";
let activeAblation = "components";
let activeCondition = "Standard";
let activeMetric = "success";
let activeVideoTask = 0;
let activeVideoVariant = 0;

function liberoMarkup() {
  const ranking = [...liberoRows]
    .sort((a, b) => b.scores[4] - a.scores[4])
    .map((row, index) => `
      <div class="rank-row${row.accent ? " accent" : ""}">
        <span class="rank-index">${String(index + 1).padStart(2, "0")}</span>
        <span class="rank-name">${row.method}</span>
        <span class="rank-track"><span style="width:${row.scores[4]}%"></span></span>
        <strong>${row.scores[4].toFixed(1)}</strong>
      </div>`)
    .join("");

  const rows = liberoRows.map((row) => `
    <tr class="${row.accent ? "highlight" : ""}">
      <th scope="row">${row.method}</th>
      ${row.scores.map((score) => `<td>${score.toFixed(1)}</td>`).join("")}
    </tr>`).join("");

  return `
    <div class="result-panel" role="tabpanel">
      <div class="panel-intro">
        <div>
          <span class="panel-kicker">40 tasks · 2,000 evaluation trials</span>
          <h3>LIBERO suite success rate</h3>
          <p>W²-VLA leads the aggregate and three of four suites, reaching a 98.5% overall average.</p>
        </div>
        <div class="big-number"><strong>+1.3</strong><span>pts over the strongest baseline</span></div>
      </div>
      <div class="split-results">
        <div class="rank-chart" aria-label="LIBERO average score comparison">${ranking}</div>
        <div class="table-wrap">
          <table>
            <thead><tr><th scope="col">Method</th><th scope="col">Spatial</th><th scope="col">Object</th><th scope="col">Goal</th><th scope="col">Long</th><th scope="col">Avg.</th></tr></thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </div>
    </div>`;
}

function robotwinMarkup() {
  const bars = robotwinRows.map((row) => `
    <div class="twin-row${row.accent ? " accent" : ""}">
      <span>${row.method}</span>
      <div class="twin-bars">
        <i class="easy" style="width:${(row.easy / 0.65).toFixed(2)}%"><b>${row.easy.toFixed(2)}</b></i>
        <i class="hard" style="width:${row.hard === null ? 0 : (row.hard / 0.2).toFixed(2)}%"><b>${row.hard === null ? "—" : row.hard.toFixed(2)}</b></i>
      </div>
    </div>`).join("");

  const rows = robotwinRows.map((row) => `
    <tr class="${row.accent ? "highlight" : ""}">
      <th scope="row">${row.method}</th>
      <td>${row.easy.toFixed(2)}</td>
      <td>${row.hard === null ? "—" : row.hard.toFixed(2)}</td>
    </tr>`).join("");

  return `
    <div class="result-panel" role="tabpanel">
      <div class="panel-intro">
        <div>
          <span class="panel-kicker">50 bimanual tasks · clean and domain-randomized</span>
          <h3>RoboTwin 2.0 success rate</h3>
          <p>The gain persists under both Easy and Hard settings, demonstrating stronger bimanual generalization.</p>
        </div>
        <div class="big-number"><strong>60.71</strong><span>Easy setting success</span></div>
      </div>
      <div class="robotwin-chart" aria-label="RoboTwin easy and hard score comparison">
        <div class="chart-legend"><span class="legend-easy">Easy</span><span class="legend-hard">Hard</span></div>
        ${bars}
      </div>
      <div class="table-wrap compact-table">
        <table><thead><tr><th scope="col">Method</th><th scope="col">Easy</th><th scope="col">Hard</th></tr></thead><tbody>${rows}</tbody></table>
      </div>
    </div>`;
}

function ablationMarkup() {
  const panel = ablationPanels[activeAblation];
  const subtabs = Object.entries(ablationPanels).map(([key, item]) => `
    <button type="button" data-ablation="${key}" class="${key === activeAblation ? "active" : ""}" aria-pressed="${key === activeAblation}">${item.label}</button>`).join("");
  const headers = panel.headers.map((header) => `<th scope="col">${header}</th>`).join("");
  const rows = panel.rows.map((row) => `
    <tr class="${row[0] === "W²-VLA" ? "highlight" : ""}">
      ${row.map((cell, index) => index === 0 ? `<th scope="row">${cell}</th>` : `<td>${cell}</td>`).join("")}
    </tr>`).join("");

  return `
    <div class="result-panel" role="tabpanel">
      <div class="panel-intro">
        <div>
          <span class="panel-kicker">LIBERO ablation study</span>
          <h3>What makes World-to-Wrist work?</h3>
          <p>Inspect component contributions, the latent interface, and the future-view prediction target.</p>
        </div>
      </div>
      <div class="subtabs" aria-label="Ablation panels">${subtabs}</div>
      <div class="ablation-callout"><span>Key readout</span><p>${panel.note}</p></div>
      <div class="table-wrap"><table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table></div>
    </div>`;
}

function renderBenchmark() {
  benchmarkPanel.innerHTML = activeBenchmark === "libero"
    ? liberoMarkup()
    : activeBenchmark === "robotwin"
      ? robotwinMarkup()
      : ablationMarkup();

  document.querySelectorAll("[data-benchmark]").forEach((button) => {
    const isActive = button.dataset.benchmark === activeBenchmark;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
}

function renderRealWorld() {
  const values = realWorld[activeCondition][activeMetric];
  const summaries = values.map((method) => method.reduce((sum, value) => sum + value, 0) / method.length);
  const clusters = taskNames.map((task, taskIndex) => {
    const bars = methodNames.map((method, methodIndex) => {
      const value = values[methodIndex][taskIndex];
      const normalized = activeMetric === "success" ? value : (value / taskMax[taskIndex]) * 100;
      return `<div class="bar-slot"><span class="bar method-${methodIndex}" style="--bar-height:${normalized}%"><b>${value.toFixed(activeMetric === "success" ? 1 : 2)}</b></span></div>`;
    }).join("");
    return `
      <div class="bar-cluster">
        <div class="bars">${bars}</div>
        <strong>${task}</strong>
        <small>${activeMetric === "progress" ? `max ${taskMax[taskIndex]}` : "30 trials / method"}</small>
      </div>`;
  }).join("");

  const legend = methodNames.map((method, index) => `<span class="method-${index}">${method}</span>`).join("");
  const summary = methodNames.map((method, index) => `
    <div class="${index === 2 ? "winner" : ""}"><span>${method}</span><strong>${summaries[index].toFixed(2)}${activeMetric === "success" ? "%" : ""}</strong></div>`).join("");

  document.querySelector("#real-content").innerHTML = `
    <div class="vertical-chart" aria-label="${activeCondition} ${activeMetric} comparison">
      <div class="chart-topline"><span>${activeMetric === "success" ? "Success rate (%)" : "Ordered progress score"}</span><span>higher is better ↑</span></div>
      <div class="chart-area">${clusters}</div>
      <div class="method-legend">${legend}</div>
    </div>
    <aside class="real-summary">
      <span class="panel-kicker">Average across tasks</span>
      <h3>${activeCondition} · ${activeMetric === "success" ? "success" : "progress"}</h3>
      <div class="summary-list">${summary}</div>
      <p>W²-VLA ranks first on every task in both Standard and OOD conditions.</p>
    </aside>`;

  document.querySelectorAll("[data-condition]").forEach((button) => {
    const isActive = button.dataset.condition === activeCondition;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
  document.querySelectorAll("[data-metric]").forEach((button) => {
    const isActive = button.dataset.metric === activeMetric;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

function updateVideo() {
  const task = videoTasks[activeVideoTask];
  const variant = videoVariants[activeVideoVariant];
  const base = "https://huggingface.co/datasets/HarrisonPENG/W2-VLA-assets";
  const player = document.querySelector("#demo-video");
  const source = document.querySelector("#demo-video-source");

  source.src = `${base}/resolve/main/${task.folder}/${variant.file}`;
  player.setAttribute("aria-label", `${task.label}, ${variant.label} demonstration`);
  document.querySelector("#demo-folder-link").href = `${base}/tree/main/${task.folder}`;
  player.load();
  player.play().catch(() => {});

  document.querySelectorAll("[data-video-task]").forEach((button) => {
    const isActive = Number(button.dataset.videoTask) === activeVideoTask;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });
  document.querySelectorAll("[data-video-variant]").forEach((button) => {
    const isActive = Number(button.dataset.videoVariant) === activeVideoVariant;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

document.querySelectorAll("[data-benchmark]").forEach((button) => {
  button.addEventListener("click", () => {
    activeBenchmark = button.dataset.benchmark;
    renderBenchmark();
  });
});

benchmarkPanel.addEventListener("click", (event) => {
  const button = event.target.closest("[data-ablation]");
  if (!button) return;
  activeAblation = button.dataset.ablation;
  renderBenchmark();
});

document.querySelectorAll("[data-condition]").forEach((button) => {
  button.addEventListener("click", () => {
    activeCondition = button.dataset.condition;
    renderRealWorld();
  });
});

document.querySelectorAll("[data-metric]").forEach((button) => {
  button.addEventListener("click", () => {
    activeMetric = button.dataset.metric;
    renderRealWorld();
  });
});

document.querySelectorAll("[data-video-task]").forEach((button) => {
  button.addEventListener("click", () => {
    activeVideoTask = Number(button.dataset.videoTask);
    updateVideo();
  });
});

document.querySelectorAll("[data-video-variant]").forEach((button) => {
  button.addEventListener("click", () => {
    activeVideoVariant = Number(button.dataset.videoVariant);
    updateVideo();
  });
});

const siteHeader = document.querySelector(".site-header");
const menuToggle = document.querySelector(".menu-toggle");

function closeMobileMenu() {
  siteHeader.classList.remove("menu-open");
  menuToggle.setAttribute("aria-expanded", "false");
  menuToggle.setAttribute("aria-label", "Open navigation");
}

menuToggle.addEventListener("click", () => {
  const isOpen = siteHeader.classList.toggle("menu-open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
  menuToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
});

siteHeader.querySelectorAll("nav a").forEach((link) => {
  link.addEventListener("click", closeMobileMenu);
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeMobileMenu();
});

renderBenchmark();
renderRealWorld();
