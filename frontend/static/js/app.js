// ─────────────────────────────────────────────────────────────────────────────
// AIRA — app.js  (GRU Forecast Edition)
// ─────────────────────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:5001";

let state = {
  profile: "normal",
  age: 21,
  exposureHours: 3,
  pollutants: {},
  currentAqi: 0,
  city: "Delhi",
  pollutantChart: null,
  forecastChart: null,
  riskChart: null,
};

// ─── AQI metadata ────────────────────────────────────────────────────────────
const AQI_META = {
  Good:         { color: "#00c896", desc: "Air quality is ideal. Enjoy outdoor activities.",        pct: 6  },
  Satisfactory: { color: "#96c800", desc: "Acceptable for most. Sensitive groups take care.",       pct: 18 },
  Moderate:     { color: "#f5a623", desc: "Sensitive groups may experience health effects.",        pct: 40 },
  Poor:         { color: "#f05030", desc: "Health effects likely. Limit prolonged exposure.",       pct: 58 },
  "Very Poor":  { color: "#c03070", desc: "Health alert. Avoid outdoor activities.",                pct: 76 },
  Severe:       { color: "#8020a0", desc: "Emergency conditions. Stay indoors immediately.",        pct: 95 },
};

const POLLUTANT_META = {
  "PM2.5": { icon: "ti-cloud-fog",    color: "#f05030", max: 300, unit: "μg/m³" },
  "PM10":  { icon: "ti-cloud",        color: "#f5a623", max: 500, unit: "μg/m³" },
  "NO2":   { icon: "ti-flask",        color: "#a070e0", max: 400, unit: "ppb"   },
  "CO":    { icon: "ti-flame",        color: "#e07050", max: 50,  unit: "ppm"   },
  "SO2":   { icon: "ti-droplet-half", color: "#70b0e0", max: 800, unit: "ppb"   },
  "O3":    { icon: "ti-sun",          color: "#00c896", max: 300, unit: "ppb"   },
};

const RISK_COLORS = {
  Low: {
    color: "#00d4aa",
    bg: "rgba(0,212,170,0.1)",
    border: "rgba(0,212,170,0.2)",
    icon: "ti-shield-check"
  },
  Moderate: {
    color: "#f5a623",
    bg: "rgba(245,166,35,0.1)",
    border: "rgba(245,166,35,0.2)",
    icon: "ti-shield-exclamation"
  },
  High: {
    color: "#e8394a",
    bg: "rgba(232,57,74,0.1)",
    border: "rgba(232,57,74,0.2)",
    icon: "ti-shield-x"
  }
};

// ─── Utilities ───────────────────────────────────────────────────────────────
function setStatus(text, live = false) {
  document.getElementById("statusText").textContent = text;
  document.getElementById("statusDot").classList.toggle("live", live);
}

function scrollToSection(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
}

function updateSlider() {
  const val = document.getElementById("exposureSlider").value;
  document.getElementById("exposureVal").textContent = val;
  state.exposureHours = parseInt(val);
}

function selectProfile(btn) {
  document.querySelectorAll(".profile-pill").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  state.profile = btn.dataset.profile;
}

function showLoader(show) {
  const lb = document.getElementById("loaderBar");
  const lf = document.getElementById("loaderFill");
  if (show) {
    lb.style.display = "block";
    setTimeout(() => { lf.style.width = "70%"; }, 50);
  } else {
    lf.style.width = "100%";
    setTimeout(() => { lb.style.display = "none"; lf.style.width = "0%"; }, 400);
  }
}

function getAqiCategory(aqi) {
  if (aqi <= 50)  return "Good";
  if (aqi <= 100) return "Satisfactory";
  if (aqi <= 200) return "Moderate";
  if (aqi <= 300) return "Poor";
  if (aqi <= 400) return "Very Poor";
  return "Severe";
}

// ─── Location & Live AQI ─────────────────────────────────────────────────────
function requestLocation() {
  const btn = document.getElementById("locateBtn");
  btn.disabled = true;
  btn.innerHTML = '<i class="ti ti-loader" aria-hidden="true"></i> Detecting…';
  setStatus("Getting location…");

  if (!navigator.geolocation) {
    alert("Geolocation is not supported by your browser.");
    btn.disabled = false;
    btn.innerHTML = '<i class="ti ti-current-location" aria-hidden="true"></i> Detect My Location';
    return;
  }

  navigator.geolocation.getCurrentPosition(
    pos => fetchAqi(pos.coords.latitude, pos.coords.longitude),
    () => {
      setStatus("Location denied");
      btn.disabled = false;
      btn.innerHTML = '<i class="ti ti-current-location" aria-hidden="true"></i> Retry Location';
      alert("Location access denied. Please allow location access and try again.");
    }
  );
}

async function fetchAqi(lat, lon) {
  showLoader(true);
  setStatus("Fetching live AQI…");

  try {
    const res  = await fetch(`${API_BASE}/api/fetch-aqi?lat=${lat}&lon=${lon}`);
    const data = await res.json();

    if (data.error) throw new Error(data.error);

    state.pollutants = {
      "PM2.5": data.pm25, "PM10": data.pm10,
      "NO2":   data.no2,  "CO":   data.co,
      "SO2":   data.so2,  "O3":   data.o3,
    };
    state.currentAqi = data.current_aqi;
    state.city = data.city || "Delhi";

    document.getElementById("locationLabel").textContent = data.city || "Your Location";
    document.getElementById("lastUpdated").textContent   = data.time
      ? `Updated ${new Date(data.time).toLocaleTimeString()}`
      : "Just now";
    document.getElementById("orbValue").textContent = data.current_aqi;

    renderAqiBanner(data.current_aqi);
    renderPollutantCards();
    renderPollutantChart();

    const ds = document.getElementById("dataSection");
    ds.style.display = "flex";
    ds.classList.add("fade-in");
    setStatus(`${data.city || "Live"}`, true);

    document.getElementById("locateBtn").innerHTML = '<i class="ti ti-refresh" aria-hidden="true"></i> Refresh Data';
    document.getElementById("locateBtn").disabled  = false;

  } catch (e) {
    setStatus("Error fetching data");
    alert("Failed to fetch AQI data: " + e.message);
    document.getElementById("locateBtn").disabled  = false;
    document.getElementById("locateBtn").innerHTML = '<i class="ti ti-current-location" aria-hidden="true"></i> Retry Location';
  }

  showLoader(false);
}

// ─── Live dashboard renderers ─────────────────────────────────────────────────
function renderAqiBanner(aqi) {
  const cat  = getAqiCategory(aqi);
  const meta = AQI_META[cat] || AQI_META.Moderate;

  document.getElementById("bannerAqi").textContent      = aqi;
  document.getElementById("bannerAqi").style.color      = meta.color;
  document.getElementById("bannerCategory").textContent = cat;
  document.getElementById("bannerCategory").style.color = meta.color;
  document.getElementById("bannerDesc").textContent     = meta.desc;
  document.getElementById("scaleThumb").style.left      = meta.pct + "%";
  document.getElementById("aqiBanner").style.borderColor= meta.color + "40";
}

function renderPollutantCards() {
  const grid = document.getElementById("pollutantGrid");
  grid.innerHTML = "";

  Object.entries(state.pollutants).forEach(([name, val]) => {
    const meta = POLLUTANT_META[name] || { icon: "ti-atom", color: "#888", max: 100, unit: "—" };
    const pct  = Math.min(100, (val / meta.max) * 100).toFixed(1);

    const card = document.createElement("div");
    card.className = "pollutant-card fade-in";
    card.innerHTML = `
      <div class="pollutant-top">
        <span class="pollutant-name">${name}</span>
        <span class="pollutant-icon" style="background:${meta.color}18;color:${meta.color};">
          <i class="ti ${meta.icon}" aria-hidden="true"></i>
        </span>
      </div>
      <div class="pollutant-value">${val ?? "—"}</div>
      <div class="pollutant-unit">${meta.unit}</div>
      <div class="pollutant-bar-bg">
        <div class="pollutant-bar-fill" style="width:${pct}%;background:${meta.color};"></div>
      </div>`;
    grid.appendChild(card);
  });
}

function renderPollutantChart() {
  const ctx    = document.getElementById("pollutantChart").getContext("2d");
  const labels = Object.keys(state.pollutants);
  const values = Object.values(state.pollutants);
  const colors = labels.map(l => POLLUTANT_META[l]?.color || "#888");

  if (state.pollutantChart) state.pollutantChart.destroy();

  state.pollutantChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Concentration",
        data:  values,
        backgroundColor: colors.map(c => c + "55"),
        borderColor:     colors,
        borderWidth:     1.5,
        borderRadius:    6,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: tooltipDefaults() },
      scales:  { x: scaleDefaults(), y: { ...scaleDefaults(), beginAtZero: true } },
    }
  });
}

// ─── GRU FORECAST ─────────────────────────────────────────────────────────────
async function runForecast() {
  const btn = document.getElementById("analyzeBtn");
  btn.disabled = true;
  btn.innerHTML = '<i class="ti ti-loader" aria-hidden="true"></i> Forecasting…';

  try {

    // Read age from input
    state.age = parseInt(document.getElementById("ageInput").value) || 21;

    const payload = {
      pm25:           state.pollutants["PM2.5"] || 0,
      pm10:           state.pollutants["PM10"]  || 0,
      no2:            state.pollutants["NO2"]   || 0,
      co:             state.pollutants["CO"]    || 0,
      so2:            state.pollutants["SO2"]   || 0,
      o3:             state.pollutants["O3"]    || 0,
      current_aqi:    state.currentAqi,
      exposure_hours: state.exposureHours,
      profile:        state.profile,
      age:            state.age,
      city:           state.city || "Delhi"
    };

    const res = await fetch(`${API_BASE}/api/forecast`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await res.json();

    if (data.error) {
      throw new Error(data.error);
    }

    renderForecast(data);

  } catch (e) {
    alert("Forecast failed: " + e.message);
  }

  btn.disabled = false;
  btn.innerHTML =
    '<i class="ti ti-brain" aria-hidden="true"></i> Generate 24h Forecast';
}

function renderForecast(data) {
  const {
    forecast_aqi, forecast_lower, forecast_upper, confidence,
    risk_timeline, peak_aqi, peak_hour,
    anomaly, recommendations, model_used, gru_metrics,
  } = data;

  // ── Model badge ──────────────────────────────────────────────────────────
  const modelBadgeText = document.getElementById("modelBadgeText");
  if (model_used === "GRU") {
    const r2 = gru_metrics?.r2 ? ` · R² ${gru_metrics.r2.toFixed(3)}` : "";
    modelBadgeText.textContent = `GRU Neural Network${r2}`;
  } else {
    modelBadgeText.textContent = "RF Fallback (run train_gru.py)";
  }

  document.getElementById("confidenceText").textContent =
    `${Math.round(confidence * 100)}% confidence`;

  // ── Anomaly alert ─────────────────────────────────────────────────────────
  const alert_ = document.getElementById("anomalyAlert");
  if (anomaly.detected) {
    document.getElementById("anomalyDesc").textContent = " — " + anomaly.description;
    alert_.style.display = "flex";
    alert_.classList.add("fade-in");
  } else {
    alert_.style.display = "none";
  }

  // ── Summary cards (6h, 12h, 24h, peak) ───────────────────────────────────
  const snapshots = [
    { id: "fc6h",  hourIdx: 5  },
    { id: "fc12h", hourIdx: 11 },
    { id: "fc24h", hourIdx: 23 },
  ];
  snapshots.forEach(({ id, hourIdx }) => {
    const entry = risk_timeline[hourIdx];
    if (!entry) return;
    const card = document.getElementById(id);
    const meta = AQI_META[entry.category] || AQI_META.Moderate;
    card.querySelector(".fc-aqi").textContent  = entry.aqi;
    card.querySelector(".fc-aqi").style.color  = meta.color;
    card.querySelector(".fc-cat").textContent  = entry.category;
    card.querySelector(".fc-risk").textContent = `Risk: ${entry.risk_level} (${entry.risk_score})`;
    card.style.borderColor = meta.color + "40";
  });

  // Peak card
  const peakMeta = AQI_META[getAqiCategory(peak_aqi)] || AQI_META.Moderate;
  document.getElementById("peakAqiVal").textContent  = peak_aqi;
  document.getElementById("peakAqiVal").style.color  = peakMeta.color;
  document.getElementById("peakHourVal").textContent = `at hour ${peak_hour}`;
  document.getElementById("peakCatVal").textContent  = getAqiCategory(peak_aqi);
  document.getElementById("fcPeak").style.borderColor= peakMeta.color + "60";

  // ── Forecast line chart ───────────────────────────────────────────────────
  renderForecastChart(forecast_aqi, forecast_lower, forecast_upper, risk_timeline);

  // ── Risk timeline bar chart ───────────────────────────────────────────────
  renderRiskTimelineChart(risk_timeline);
  // ── User summary card ────────────────────────────────────────────────────
  document.getElementById("profileInfo").textContent =
    state.profile.charAt(0).toUpperCase() + state.profile.slice(1);
  
  document.getElementById("ageInfo").textContent =
    state.age;

  document.getElementById("exposureInfo").textContent =
    `${state.exposureHours} hrs/day`;
  
  // ── Recommendations ───────────────────────────────────────────────────────
  const recoList = document.getElementById("recoList");
  recoList.innerHTML = "";
  recommendations.forEach((r, i) => {
    const item = document.createElement("div");
    item.className = "reco-item";
    item.style.animationDelay = `${i * 0.08}s`;
    const pm = {
      low:    { color: "#00c896", bg: "rgba(0,200,150,0.1)",  label: "Safe"    },
      medium: { color: "#f5a623", bg: "rgba(245,166,35,0.1)", label: "Caution" },
      high:   { color: "#e8394a", bg: "rgba(232,57,74,0.1)",  label: "Action"  },
    }[r.priority] || { color: "#f5a623", bg: "rgba(245,166,35,0.1)", label: "Caution" };

    item.innerHTML = `
      <div class="reco-icon" style="background:${pm.bg};color:${pm.color};">
        <i class="ti ${r.icon || "ti-info-circle"}" aria-hidden="true"></i>
      </div>
      <span class="reco-text">${r.text}</span>
      <span class="reco-badge" style="background:${pm.bg};color:${pm.color};">${pm.label}</span>`;
    recoList.appendChild(item);
  });

  // ── Risk gauge (peak risk) ────────────────────────────────────────────────
  const peakRiskEntry = risk_timeline[peak_hour - 1] || risk_timeline[risk_timeline.length - 1];
  if (peakRiskEntry) {
    drawGauge(peakRiskEntry.risk_score);
    document.getElementById("gaugeLabel").textContent =
      `Peak Risk Score: ${peakRiskEntry.risk_score.toFixed(2)} / 10`;
  }

  // ── Show forecast section ─────────────────────────────────────────────────
  const fs = document.getElementById("forecast");
  fs.style.display = "flex";
  fs.classList.add("fade-in");
  fs.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── Chart: 24h Forecast Line ─────────────────────────────────────────────────
function renderForecastChart(forecast_aqi, lower, upper, risk_timeline) {
  const ctx = document.getElementById("forecastChart").getContext("2d");
  if (state.forecastChart) state.forecastChart.destroy();

  const labels    = Array.from({ length: 24 }, (_, i) => `+${i + 1}h`);
  const lineColors = forecast_aqi.map(v => AQI_META[getAqiCategory(v)]?.color || "#888");

  state.forecastChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label:           "Forecast AQI",
          data:            forecast_aqi,
          borderColor:     "var(--accent)",
          backgroundColor: "rgba(0,212,170,0.08)",
          borderWidth:     2.5,
          pointRadius:     3,
          pointBackgroundColor: lineColors,
          tension:         0.4,
          fill:            false,
          order:           1,
        },
        {
          label:           "Upper 90%",
          data:            upper,
          borderColor:     "rgba(255,255,255,0.1)",
          backgroundColor: "rgba(0,212,170,0.06)",
          borderWidth:     1,
          borderDash:      [4, 4],
          pointRadius:     0,
          tension:         0.4,
          fill:            "+1",
          order:           2,
        },
        {
          label:           "Lower 90%",
          data:            lower,
          borderColor:     "rgba(255,255,255,0.1)",
          backgroundColor: "rgba(0,212,170,0.06)",
          borderWidth:     1,
          borderDash:      [4, 4],
          pointRadius:     0,
          tension:         0.4,
          fill:            false,
          order:           3,
        },
      ]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      interaction:         { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...tooltipDefaults(),
          callbacks: {
            label: ctx => {
              if (ctx.datasetIndex === 0) {
                const cat = getAqiCategory(ctx.parsed.y);
                return ` AQI ${ctx.parsed.y} — ${cat}`;
              }
              if (ctx.datasetIndex === 1) return ` Upper: ${ctx.parsed.y}`;
              return ` Lower: ${ctx.parsed.y}`;
            }
          }
        }
      },
      scales: {
        x: { ...scaleDefaults() },
        y: { ...scaleDefaults(), beginAtZero: true,
             title: { display: true, text: "AQI", color: "rgba(160,180,200,0.6)" } },
      }
    }
  });
}

// ─── Chart: Risk Timeline Bar ─────────────────────────────────────────────────
function renderRiskTimelineChart(risk_timeline) {
  const ctx = document.getElementById("riskTimelineChart").getContext("2d");
  if (state.riskChart) state.riskChart.destroy();

  const labels     = risk_timeline.map(r => `+${r.hour}h`);
  const scores     = risk_timeline.map(r => r.risk_score);
  const bgColors   = risk_timeline.map(r => {
    const meta = RISK_COLORS[r.risk_level] || RISK_COLORS.Low;
    return meta.color + "88";
  });
  const bdColors   = risk_timeline.map(r => {
    const meta = RISK_COLORS[r.risk_level] || RISK_COLORS.Low;
    return meta.color;
  });

  state.riskChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label:           "Risk Score",
        data:            scores,
        backgroundColor: bgColors,
        borderColor:     bdColors,
        borderWidth:     1,
        borderRadius:    3,
      }]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...tooltipDefaults(),
          callbacks: {
            label: ctx => {
              const entry = risk_timeline[ctx.dataIndex];
              return ` ${entry.risk_level} risk — score ${entry.risk_score}`;
            }
          }
        }
      },
      scales: {
        x: { ...scaleDefaults(), ticks: { ...scaleDefaults().ticks, maxTicksLimit: 12 } },
        y: { ...scaleDefaults(), beginAtZero: true,
             title: { display: true, text: "Risk Score", color: "rgba(160,180,200,0.6)" } },
      }
    }
  });
}

// ─── Gauge ────────────────────────────────────────────────────────────────────
function drawGauge(score) {
  const canvas = document.getElementById("riskGauge");
  const ctx    = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 20, r = 110;
  const startAngle = Math.PI, endAngle = 2 * Math.PI;

  const arcGrad = ctx.createLinearGradient(cx - r, 0, cx + r, 0);
  arcGrad.addColorStop(0,   "#00c896");
  arcGrad.addColorStop(0.4, "#f5a623");
  arcGrad.addColorStop(1,   "#e8394a");

  // Background arc
  ctx.beginPath(); ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = "rgba(255,255,255,0.06)"; ctx.lineWidth = 16; ctx.lineCap = "round"; ctx.stroke();

  // Filled arc
  const maxScore  = 10;
  const fillAngle = startAngle + (Math.min(score, maxScore) / maxScore) * Math.PI;
  ctx.beginPath(); ctx.arc(cx, cy, r, startAngle, fillAngle);
  ctx.strokeStyle = arcGrad; ctx.lineWidth = 16; ctx.lineCap = "round"; ctx.stroke();

  // Knob
  const nx = cx + r * Math.cos(fillAngle), ny = cy + r * Math.sin(fillAngle);
  ctx.beginPath(); ctx.arc(nx, ny, 8, 0, 2 * Math.PI);
  ctx.fillStyle = "#ffffff"; ctx.fill();

  // Score text
  ctx.fillStyle = "rgba(255,255,255,0.9)"; ctx.font = "bold 32px 'Syne', sans-serif";
  ctx.textAlign = "center"; ctx.fillText(score.toFixed(1), cx, cy - 16);
  ctx.fillStyle = "rgba(160,180,200,0.7)"; ctx.font = "12px 'DM Sans', sans-serif";
  ctx.fillText("RISK SCORE", cx, cy + 4);
  ctx.fillStyle = "rgba(100,130,150,0.6)"; ctx.font = "11px 'DM Sans', sans-serif";
  ctx.textAlign = "left";  ctx.fillText("Low",  cx - r - 10, cy + 20);
  ctx.textAlign = "right"; ctx.fillText("High", cx + r + 10, cy + 20);
}

// ─── Chart.js shared defaults ─────────────────────────────────────────────────
function tooltipDefaults() {
  return {
    backgroundColor: "#1a2230",
    borderColor:     "rgba(255,255,255,0.1)",
    borderWidth:     1,
    titleColor:      "#f0f4f8",
    bodyColor:       "#a8b8c8",
    cornerRadius:    8,
    padding:         10,
  };
}
function scaleDefaults() {
  return {
    grid:  { color: "rgba(255,255,255,0.05)" },
    ticks: { color: "#6a7d90", font: { family: "'DM Sans'", size: 11 } },
  };
}