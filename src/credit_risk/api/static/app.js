"use strict";

const MONTHS = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"];
const SVG_NS = "http://www.w3.org/2000/svg";

const PRESETS = {
  reliable: {
    limit_bal: 200000, sex: 2, education: 2, marriage: 1, age: 38,
    pay_1: -1, pay_2: -1, pay_3: -1, pay_4: -1, pay_5: -1, pay_6: -1,
    bill_amt1: 8000, bill_amt2: 7500, bill_amt3: 9000, bill_amt4: 8200, bill_amt5: 7800, bill_amt6: 8100,
    pay_amt1: 8000, pay_amt2: 7500, pay_amt3: 9000, pay_amt4: 8200, pay_amt5: 7800, pay_amt6: 8100,
  },
  mixed: {
    limit_bal: 80000, sex: 1, education: 2, marriage: 2, age: 31,
    pay_1: 0, pay_2: 0, pay_3: 1, pay_4: 0, pay_5: 0, pay_6: -1,
    bill_amt1: 52000, bill_amt2: 49000, bill_amt3: 47000, bill_amt4: 43000, bill_amt5: 40000, bill_amt6: 38000,
    pay_amt1: 2500, pay_amt2: 2200, pay_amt3: 0, pay_amt4: 2000, pay_amt5: 1800, pay_amt6: 2000,
  },
  delays: {
    limit_bal: 30000, sex: 1, education: 3, marriage: 2, age: 27,
    pay_1: 2, pay_2: 2, pay_3: 2, pay_4: 1, pay_5: 0, pay_6: 0,
    bill_amt1: 29500, bill_amt2: 28900, bill_amt3: 28000, bill_amt4: 27000, bill_amt5: 25500, bill_amt6: 24000,
    pay_amt1: 0, pay_amt2: 1500, pay_amt3: 0, pay_amt4: 1200, pay_amt5: 1000, pay_amt6: 1000,
  },
};

const PAY_LABELS = {
  "-2": "No consumption", "-1": "Paid in full", "0": "Revolving credit", "1": "1 month late", "2": "2 months late",
  "3": "3 months late", "4": "4 months late", "5": "5 months late", "6": "6 months late",
  "7": "7 months late", "8": "8 months late", "9": "9 months late",
};

const state = { schema: {}, fields: [] };

function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) node.append(c instanceof Node ? c : document.createTextNode(c));
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  children.flat().forEach((c) => node.append(c));
  return node;
}

const fmt = {
  pct: (x, d = 1) => `${(x * 100).toFixed(d)}%`,
  num: (x, d = 3) => Number(x).toFixed(d),
};

function titleCase(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

/* ---------- form ---------- */

const FALLBACK = {
  SEX: { values: { 1: "male", 2: "female" } },
  EDUCATION: { values: { 1: "graduate school", 2: "university", 3: "high school", 4: "other" } },
  MARRIAGE: { values: { 1: "married", 2: "single", 3: "other" } },
  AGE: { min: 18, max: 100 },
};

function schemaFor(name) {
  const key = name.toUpperCase();
  return state.schema[key] || FALLBACK[key] || {};
}

function numberField(name, label, { hint, min, max, step = 1 } = {}) {
  const meta = schemaFor(name);
  const lo = min ?? meta.min, hi = max ?? meta.max;
  return { name, label, hint, kind: "number", min: lo, max: hi, step };
}

function buildFields() {
  const sel = (name, label) => ({
    name, label, kind: "select",
    options: Object.entries(schemaFor(name).values || {}).map(([v, text]) => [v, titleCase(text)]),
  });
  return [
    { title: "Profile", fields: [
      numberField("limit_bal", "Credit limit (NT$)", { min: 1 }),
      sel("sex", "Sex"), sel("education", "Education"), sel("marriage", "Marital status"),
      numberField("age", "Age (years)"),
    ] },
    { title: "Repayment history", hint: "Status per month, most recent first.", fields: MONTHS.map((m, i) => ({
      name: `pay_${i + 1}`, label: `${m} 2005`, kind: "select",
      options: Object.entries(PAY_LABELS).map(([v, t]) => [v, `${v}: ${t}`]),
    })) },
    { title: "Bill statements (NT$)", fields: MONTHS.map((m, i) => numberField(`bill_amt${i + 1}`, `${m} 2005`, { hint: i === 0 ? "negative = credit balance" : undefined })) },
    { title: "Payments made (NT$)", fields: MONTHS.map((m, i) => numberField(`pay_amt${i + 1}`, `${m} 2005`, { min: 0 })) },
  ];
}

function renderForm() {
  const root = document.getElementById("form-sections");
  root.replaceChildren();
  state.fields = [];
  for (const section of buildFields()) {
    const grid = h("div", { class: "fields" });
    for (const f of section.fields) {
      state.fields.push(f);
      const id = `f-${f.name}`;
      const control = f.kind === "select"
        ? h("select", { id, name: f.name, required: true, "aria-describedby": `${id}-err` },
            ...f.options.map(([v, t]) => h("option", { value: v, text: t })))
        : h("input", { id, name: f.name, type: "number", inputmode: "numeric", required: true, min: f.min, max: f.max, step: f.step, "aria-describedby": `${id}-err` });
      grid.append(h("div", { class: "field" },
        h("label", { for: id, text: f.label }), control,
        f.hint ? h("span", { class: "hint", text: f.hint }) : null,
        h("span", { class: "error-msg", id: `${id}-err` })));
    }
    root.append(h("fieldset", {}, h("legend", { text: section.title }), section.hint ? h("p", { class: "muted", text: section.hint }) : null, grid));
  }
  setValues(PRESETS.mixed);
}

function setValues(values) {
  for (const f of state.fields) {
    const el = document.getElementById(`f-${f.name}`);
    el.value = values && values[f.name] != null ? String(values[f.name]) : "";
  }
  clearErrors();
}

function clearErrors() {
  document.querySelectorAll("#applicant-form [aria-invalid]").forEach((e) => e.removeAttribute("aria-invalid"));
  document.querySelectorAll(".error-msg").forEach((e) => { e.textContent = ""; });
}

function setFieldError(name, message) {
  const el = document.getElementById(`f-${name}`);
  const msg = document.getElementById(`f-${name}-err`);
  if (!el || !msg) return false;
  el.setAttribute("aria-invalid", "true");
  msg.textContent = message;
  return true;
}

function readPayload() {
  const body = {};
  const missing = [];
  for (const f of state.fields) {
    const el = document.getElementById(`f-${f.name}`);
    if (el.value === "") { missing.push(f.name); continue; }
    body[f.name] = Number(el.value);
  }
  return { body, missing };
}

/* ---------- result ---------- */

function gauge(p, threshold, band) {
  const cx = 100, cy = 100, r = 84;
  const point = (t, rad) => [cx + rad * Math.cos(Math.PI * (1 - t)), cy - rad * Math.sin(Math.PI * (1 - t))];
  const arc = "M 16 100 A 84 84 0 0 1 184 100";
  const [x1, y1] = point(threshold, r - 12), [x2, y2] = point(threshold, r + 10);
  return svg("svg", { class: "gauge", viewBox: "0 0 200 112", role: "img", "aria-label": `Default probability ${fmt.pct(p)}; decision threshold ${fmt.pct(threshold)}` },
    svg("path", { d: arc, fill: "none", stroke: "var(--track)", "stroke-width": 16, "stroke-linecap": "round", pathLength: 100 }),
    svg("path", { d: arc, fill: "none", stroke: `var(--${band})`, "stroke-width": 16, "stroke-linecap": "round", pathLength: 100, "stroke-dasharray": `${Math.max(p * 100, 0.8)} 100` }),
    svg("line", { x1, y1, x2, y2, stroke: "var(--text)", "stroke-width": 2 }),
    svg("text", { x: 16, y: 110, "font-size": 10, fill: "var(--muted)", "text-anchor": "middle" }, "0%"),
    svg("text", { x: 184, y: 110, "font-size": 10, fill: "var(--muted)", "text-anchor": "middle" }, "100%"));
}

function factorList(items, unit) {
  const max = Math.max(...items.map((i) => Math.abs(i.contribution)), 1e-9);
  return h("div", {},
    h("h3", { text: "Main factors" }),
    h("ul", { class: "factors" }, ...items.map((i) => {
      const width = `${(Math.abs(i.contribution) / max) * 50}%`;
      const up = i.contribution > 0;
      return h("li", { class: "factor" },
        h("div", { class: "name" }, h("span", { text: i.label }), h("span", { class: "val", text: String(typeof i.value === "number" ? Number(i.value.toFixed(3)) : i.value) })),
        h("div", { class: "bar-track", role: "img", "aria-label": `${i.label}: ${i.direction}` },
          h("div", { class: `bar ${up ? "up" : "down"}`, style: `width:${width}` })));
    })),
    h("div", { class: "legend" }, h("span", { text: "← lowers risk" }), h("span", { text: "raises risk →" })),
    h("p", { class: "note", text: `Bar length is proportional to the contribution to the model output (${unit || "model units"}).` }));
}

function renderResult(data) {
  const out = document.getElementById("result");
  const band = data.risk_band;
  out.replaceChildren(
    gauge(data.default_probability, data.threshold, band),
    h("div", { class: "gauge-value" },
      h("div", { class: "pct", text: fmt.pct(data.default_probability) }),
      h("div", { class: "sub", text: "estimated default probability" })),
    h("div", { class: "badges" },
      h("span", { class: `badge ${data.flagged ? "high" : "low"}`, text: titleCase(data.decision) }),
      h("span", { class: `badge ${band}`, text: `${titleCase(band)} risk band` }),
      h("span", { class: "badge neutral", text: `Threshold ${fmt.pct(data.threshold)}` })),
    data.explanation ? factorList(data.explanation, data.explanation_unit) : null,
    h("p", { class: "result-disclaimer", text: "Educational demonstration on 2005 Taiwanese data. Not for real lending decisions." }));
}

function renderState(kind, title, detail, items) {
  const out = document.getElementById("result");
  const children = [];
  if (kind === "loading") children.push(h("div", { class: "spinner", "aria-hidden": "true" }));
  children.push(h("p", { text: title }));
  if (detail) children.push(h("p", { class: "muted", text: detail }));
  if (items && items.length) children.push(h("ul", {}, ...items.map((t) => h("li", { text: t }))));
  out.replaceChildren(h("div", { class: `state ${kind}`, role: kind === "error" ? "alert" : null }, ...children));
}

async function submit(event) {
  event.preventDefault();
  clearErrors();
  const { body, missing } = readPayload();
  if (missing.length) {
    missing.forEach((n) => setFieldError(n, "Required"));
    renderState("error", "Some fields are empty.", "Complete the highlighted fields.");
    document.getElementById(`f-${missing[0]}`).focus();
    return;
  }
  const button = document.getElementById("submit");
  button.disabled = true;
  renderState("loading", "Estimating…");
  try {
    const res = await fetch("/predict?explain=true&top_k=6", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { showApiError(res.status, data); return; }
    renderResult(data);
    if (window.matchMedia("(max-width: 880px)").matches) document.getElementById("result-title").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    renderState("error", "The service could not be reached.", "Check the connection and try again.");
  } finally {
    button.disabled = false;
  }
}

function showApiError(status, data) {
  const error = data.error || {};
  const unmatched = [];
  let first = null;
  for (const d of error.details || []) {
    const name = (d.field || "").split(".")[0];
    if (setFieldError(name, d.message)) first = first || name;
    else unmatched.push(d.message);
  }
  if (first) document.getElementById(`f-${first}`).focus();
  const title = status === 503 ? "The model is not available." : error.message || "The request failed.";
  renderState("error", title, status === 422 && first ? "Check the highlighted fields." : null, unmatched);
}

/* ---------- model tab ---------- */

function stat(label, value, ci) {
  return h("div", { class: "stat" }, h("div", { class: "k", text: label }), h("div", { class: "v", text: value }),
    ci ? h("div", { class: "ci", text: `95% CI ${ci[0].toFixed(3)} to ${ci[1].toFixed(3)}` }) : null);
}

function calibrationChart(curve) {
  const W = 360, H = 320, m = { l: 44, r: 14, t: 12, b: 40 };
  const maxV = Math.max(0.5, ...curve.mean_predicted, ...curve.observed_rate) * 1.08;
  const x = (v) => m.l + (v / maxV) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / maxV) * (H - m.t - m.b);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((t) => +(t * maxV).toFixed(2));
  const kids = [
    svg("line", { class: "axis", x1: m.l, y1: H - m.b, x2: W - m.r, y2: H - m.b }),
    svg("line", { class: "axis", x1: m.l, y1: m.t, x2: m.l, y2: H - m.b }),
    svg("line", { class: "diag", x1: x(0), y1: y(0), x2: x(maxV), y2: y(maxV) }),
  ];
  for (const t of ticks) {
    kids.push(svg("text", { x: x(t), y: H - m.b + 16, "text-anchor": "middle" }, t.toFixed(2)));
    kids.push(svg("text", { x: m.l - 6, y: y(t) + 4, "text-anchor": "end" }, t.toFixed(2)));
  }
  kids.push(svg("text", { x: (W + m.l) / 2, y: H - 6, "text-anchor": "middle" }, "Mean predicted probability"));
  const label = svg("text", { x: 12, y: (H - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${(H - m.b) / 2})` }, "Observed default rate");
  kids.push(label);
  const pts = curve.mean_predicted.map((p, i) => [x(p), y(curve.observed_rate[i])]);
  kids.push(svg("polyline", { class: "line", points: pts.map((p) => p.join(",")).join(" ") }));
  pts.forEach(([cx, cy]) => kids.push(svg("circle", { class: "dot", cx, cy, r: 3.5 })));
  return svg("svg", { class: "chart", viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Reliability diagram: observed default rate against predicted probability on the hold-out set" }, ...kids);
}

function renderModel(info) {
  const m = info.metrics_test || {}, ci = info.metrics_test_ci95 || {}, cost = info.cost_assumption || {};
  const root = document.getElementById("model-content");
  const ratio = cost.false_positive ? cost.false_negative / cost.false_positive : null;
  root.replaceChildren(
    h("div", { class: "grid" },
      stat("ROC-AUC", fmt.num(m.roc_auc), ci.roc_auc), stat("PR-AUC", fmt.num(m.pr_auc), ci.pr_auc),
      stat("Brier score", fmt.num(m.brier, 4), ci.brier), stat("Log loss", fmt.num(m.log_loss, 4), ci.log_loss),
      stat("Precision at threshold", fmt.num(m.precision), ci.precision), stat("Recall at threshold", fmt.num(m.recall), ci.recall)),
    h("div", { class: "two" },
      h("section", { class: "card" }, h("h2", { text: "Decision rule" }),
        h("dl", { class: "kv" },
          h("dt", { text: "Threshold" }), h("dd", { text: fmt.pct(info.threshold, 1) }),
          h("dt", { text: "Cost assumption" }), h("dd", { text: ratio ? `False negative = ${ratio} × false positive` : "n/a" }),
          h("dt", { text: "Cost per client" }), h("dd", { text: info.expected_cost_per_client_test != null ? fmt.num(info.expected_cost_per_client_test, 4) : "n/a" }),
          h("dt", { text: "Model" }), h("dd", { text: info.calibration === "raw" ? `${info.model}, no extra calibration step` : `${info.model}, ${info.calibration} calibration` }),
          h("dt", { text: "Version" }), h("dd", { text: String(info.version) }),
          h("dt", { text: "Training rows" }), h("dd", { text: info.n_train != null ? info.n_train.toLocaleString() : "n/a" }),
          h("dt", { text: "Test rows" }), h("dd", { text: info.n_test != null ? info.n_test.toLocaleString() : "n/a" })),
        h("p", { class: "note", text: (cost.note || "") + " Metrics are from the held-out test set." })),
      h("section", { class: "card" }, h("h2", { text: "Calibration (test set)" }),
        info.calibration_curve_test ? calibrationChart(info.calibration_curve_test) : h("p", { class: "muted", text: "No calibration data." }),
        h("p", { class: "note", text: "Points on the dashed diagonal mean predicted probabilities match observed default rates." }))));
}

async function loadModel() {
  const root = document.getElementById("model-content");
  try {
    const res = await fetch("/model-info");
    if (!res.ok) throw new Error(String(res.status));
    const info = await res.json();
    state.schema = Object.fromEntries((info.feature_schema || []).map((f) => [f.name, f]));
    renderModel(info);
    return info;
  } catch (err) {
    root.replaceChildren(h("div", { class: "state error", role: "alert" }, h("p", { text: "Model details are not available." }), h("p", { class: "muted", text: "The model may not be loaded." })));
    return null;
  }
}

/* ---------- tabs ---------- */

function initTabs() {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const select = (tab) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.setAttribute("aria-selected", String(on));
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    tab.focus();
  };
  tabs.forEach((t, i) => {
    t.addEventListener("click", () => select(t));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") select(tabs[(i + 1) % tabs.length]);
      if (e.key === "ArrowLeft") select(tabs[(i - 1 + tabs.length) % tabs.length]);
    });
  });
}

async function init() {
  initTabs();
  await loadModel();
  renderForm();
  document.getElementById("applicant-form").addEventListener("submit", submit);
  document.getElementById("reset").addEventListener("click", () => {
    setValues(null);
    renderState("empty", "No estimate yet.", "Fill in the form or choose an example, then select Estimate risk.");
  });
  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    setValues(PRESETS[b.dataset.preset]);
    renderState("empty", "Example loaded.", "Select Estimate risk to score it.");
  }));
}

init();
