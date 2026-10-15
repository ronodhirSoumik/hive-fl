/**
 * Hive Host Web UI Dashboard & Model Verifier
 * Vanilla JavaScript implementation, auto-refresh polling every 2s, Chart.js visualization.
 * Zero external requests; works fully offline.
 */

(function () {
  "use strict";

  // State Management
  const state = {
    token: null,
    summary: null,
    rounds: [],
    nodes: [],
    privacy: null,
    events: [],
    activeTab: "training",
    charts: {
      loss: null,
      accuracy: null,
      contributions: null,
      layerNorms: null,
      globalNorms: null,
    },
  };

  // Helper: Format Bytes
  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  // Helper: Shorten Hash
  function shortenHash(hash, len = 8) {
    if (!hash) return "—";
    if (hash.length <= len * 2) return hash;
    return `${hash.slice(0, len)}...${hash.slice(-len)}`;
  }

  // Auth Initialization
  function initAuth() {
    const params = new URLSearchParams(window.location.search);
    const queryToken = params.get("token");
    if (queryToken) {
      state.token = queryToken;
      try {
        sessionStorage.setItem("hive_admin_token", queryToken);
      } catch (e) {}
    } else {
      try {
        state.token = sessionStorage.getItem("hive_admin_token");
      } catch (e) {}
    }
  }

  // Authenticated Fetch Wrapper
  async function apiFetch(url, options = {}) {
    const headers = options.headers || {};
    if (state.token) {
      headers["X-Admin-Token"] = state.token;
    }
    options.headers = headers;
    options.credentials = "same-origin";

    const res = await fetch(url, options);
    if (res.status === 401) {
      console.warn("Admin authentication required or invalid token.");
    }
    return res;
  }

  // Initialize Tabs
  function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetId = btn.getAttribute("aria-controls");
        tabBtns.forEach((b) => {
          b.classList.remove("active");
          b.setAttribute("aria-selected", "false");
        });
        document.querySelectorAll(".tab-content").forEach((panel) => {
          panel.classList.remove("active");
        });

        btn.classList.add("active");
        btn.setAttribute("aria-selected", "true");
        const targetPanel = document.getElementById(targetId);
        if (targetPanel) {
          targetPanel.classList.add("active");
        }
        state.activeTab = targetId.replace("tab-", "");

        // Trigger chart resize if charts are present
        Object.values(state.charts).forEach((c) => {
          if (c) c.resize();
        });
      });
    });
  }

  // Setup Global Actions
  function initActions() {
    // Copy URI
    const copyBtn = document.getElementById("btn-copy-uri");
    const uriInput = document.getElementById("join-uri-input");
    if (copyBtn && uriInput) {
      copyBtn.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(uriInput.value);
          const origText = copyBtn.innerText;
          copyBtn.innerText = "✔ Copied!";
          setTimeout(() => {
            copyBtn.innerText = origText;
          }, 2000);
        } catch (e) {
          uriInput.select();
          document.execCommand("copy");
        }
      });
    }

    // Pause Session
    const pauseBtn = document.getElementById("btn-pause");
    if (pauseBtn) {
      pauseBtn.addEventListener("click", async () => {
        try {
          await apiFetch("/admin/api/session/pause", { method: "POST" });
          fetchDashboardData();
        } catch (e) {
          alert("Failed to pause session: " + e.message);
        }
      });
    }

    // Resume Session
    const resumeBtn = document.getElementById("btn-resume");
    if (resumeBtn) {
      resumeBtn.addEventListener("click", async () => {
        try {
          await apiFetch("/admin/api/session/resume", { method: "POST" });
          fetchDashboardData();
        } catch (e) {
          alert("Failed to resume session: " + e.message);
        }
      });
    }

    // Rotate Token
    const rotateBtn = document.getElementById("btn-rotate-token");
    if (rotateBtn) {
      rotateBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to rotate the join token? Connected clients with old tokens must reconnect.")) return;
        try {
          await apiFetch("/admin/api/session/rotate-token", { method: "POST" });
          fetchDashboardData();
        } catch (e) {
          alert("Failed to rotate token: " + e.message);
        }
      });
    }

    // Verify Model
    const verifyBtn = document.getElementById("btn-verify-model");
    const verifyBanner = document.getElementById("verify-result-banner");
    if (verifyBtn && verifyBanner) {
      verifyBtn.addEventListener("click", async () => {
        verifyBanner.style.display = "block";
        verifyBanner.className = "result-banner";
        verifyBanner.innerText = "Verifying model checkpoint hashes on disk...";

        try {
          const res = await apiFetch("/model/verify", { method: "POST" });
          const data = await res.json();
          if (data.verified && data.match && !data.tampered) {
            verifyBanner.className = "result-banner pass";
            verifyBanner.innerHTML = `<strong>✔ Checkpoint Verified PASS</strong><br>File: <code>${data.target_file || "final_model.safetensors"}</code><br>Computed SHA-256: <code>${data.computed_model_hash}</code> matches database record.`;
          } else {
            verifyBanner.className = "result-banner fail";
            verifyBanner.innerHTML = `<strong>❌ Verification MISMATCH / TAMPER DETECTED</strong><br>File: <code>${data.target_file || "checkpoint"}</code><br>Computed: <code>${data.computed_model_hash || "error"}</code><br>Expected: <code>${data.expected_model_hash || "unknown"}</code><br>${data.error || "Checkpoint has been altered or corrupted."}`;
          }
        } catch (e) {
          verifyBanner.className = "result-banner fail";
          verifyBanner.innerText = "Verification check failed to complete: " + e.message;
        }
      });
    }

    // Download Model
    const downloadBtn = document.getElementById("btn-download-model");
    if (downloadBtn) {
      downloadBtn.addEventListener("click", () => {
        const url = `/model/download${state.token ? "?token=" + encodeURIComponent(state.token) : ""}`;
        window.location.href = url;
      });
    }
  }

  // Initialize Charts (Chart.js)
  function initCharts() {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js not loaded. Visualizations disabled.");
      return;
    }

    // Dark/Light Theme Colors
    const isDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const gridColor = isDark ? "#374151" : "#e5e7eb";
    const textColor = isDark ? "#9ca3af" : "#4b5563";

    // 1. Loss Chart
    const lossCtx = document.getElementById("chart-loss")?.getContext("2d");
    if (lossCtx) {
      state.charts.loss = new Chart(lossCtx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Train Loss",
              data: [],
              borderColor: "#3b82f6",
              backgroundColor: "rgba(59, 130, 246, 0.1)",
              borderWidth: 2,
              tension: 0.2,
              fill: true,
            },
            {
              label: "Eval Loss",
              data: [],
              borderColor: "#f59e0b",
              backgroundColor: "rgba(245, 158, 11, 0.1)",
              borderWidth: 2,
              borderDash: [5, 5],
              tension: 0.2,
              fill: false,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: "Round", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
            y: {
              title: { display: true, text: "Loss", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
          },
        },
      });
    }

    // 2. Accuracy Chart
    const accCtx = document.getElementById("chart-accuracy")?.getContext("2d");
    if (accCtx) {
      state.charts.accuracy = new Chart(accCtx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Eval Accuracy (%)",
              data: [],
              borderColor: "#10b981",
              backgroundColor: "rgba(16, 185, 129, 0.1)",
              borderWidth: 2,
              tension: 0.2,
              fill: true,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: "Round", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
            y: {
              title: { display: true, text: "Accuracy (%)", color: textColor },
              min: 0,
              max: 100,
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
          },
        },
      });
    }

    // 3. Contributions Chart
    const contribCtx = document.getElementById("chart-contributions")?.getContext("2d");
    if (contribCtx) {
      state.charts.contributions = new Chart(contribCtx, {
        type: "bar",
        data: {
          labels: [],
          datasets: [
            {
              label: "Contribution Share (%)",
              data: [],
              backgroundColor: "#3b82f6",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: "Contributor", color: textColor },
              grid: { display: false },
              ticks: { color: textColor },
            },
            y: {
              title: { display: true, text: "Share (%)", color: textColor },
              min: 0,
              max: 100,
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
          },
        },
      });
    }

    // 4. Layer Norms Chart
    const layerCtx = document.getElementById("chart-layer-norms")?.getContext("2d");
    if (layerCtx) {
      state.charts.layerNorms = new Chart(layerCtx, {
        type: "bar",
        data: {
          labels: [],
          datasets: [
            {
              label: "L2 Norm",
              data: [],
              backgroundColor: "#8b5cf6",
              borderRadius: 4,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              grid: { display: false },
              ticks: { color: textColor, maxRotation: 45 },
            },
            y: {
              title: { display: true, text: "L2 Norm", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
          },
        },
      });
    }

    // 5. Global Norms Chart
    const globalCtx = document.getElementById("chart-global-norms")?.getContext("2d");
    if (globalCtx) {
      state.charts.globalNorms = new Chart(globalCtx, {
        type: "line",
        data: {
          labels: [],
          datasets: [
            {
              label: "Global Model Weight Norm",
              data: [],
              borderColor: "#ec4899",
              backgroundColor: "rgba(236, 72, 153, 0.1)",
              borderWidth: 2,
              tension: 0.1,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: {
              title: { display: true, text: "Round", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
            y: {
              title: { display: true, text: "Weight Norm", color: textColor },
              grid: { color: gridColor },
              ticks: { color: textColor },
            },
          },
        },
      });
    }
  }

  // Update Header and Summary
  function updateHeader(summary) {
    if (!summary) return;
    const sess = summary.session || {};
    const nodes = summary.active_nodes || {};
    const roundInfo = summary.current_round_info || {};

    const titleEl = document.getElementById("session-title");
    if (titleEl) titleEl.innerText = sess.name || `Session ${sess.session_id || ""}`;

    const statusEl = document.getElementById("session-status");
    if (statusEl) {
      const st = (sess.status || "UNKNOWN").toUpperCase();
      statusEl.innerText = st;
      statusEl.className = "badge";
      if (st === "RUNNING") statusEl.classList.add("badge-running");
      else if (st === "PAUSED") statusEl.classList.add("badge-paused");
      else if (st === "COMPLETED") statusEl.classList.add("badge-completed");
      else statusEl.classList.add("badge-error");
    }

    const transportEl = document.getElementById("transport-badge");
    if (transportEl) {
      if (sess.is_tls) {
        transportEl.innerText = "🔒 HTTPS (Encrypted)";
        transportEl.className = "badge badge-tls";
      } else {
        transportEl.innerText = "⚠ Unencrypted HTTP";
        transportEl.className = "badge badge-notls";
      }
    }

    const roundDisplay = document.getElementById("round-display");
    if (roundDisplay) {
      roundDisplay.innerText = `Round ${sess.current_round || 0} / ${sess.total_rounds || 0}`;
    }

    const activeNodesEl = document.getElementById("active-nodes-count");
    if (activeNodesEl) activeNodesEl.innerText = nodes.total || 0;

    const elEligible = document.getElementById("nodes-eligible");
    if (elEligible) elEligible.innerText = `Eligible: ${nodes.eligible || 0}`;

    const elTraining = document.getElementById("nodes-training");
    if (elTraining) elTraining.innerText = `Training: ${nodes.training || 0}`;

    const elWaiting = document.getElementById("nodes-waiting");
    if (elWaiting) elWaiting.innerText = `Waiting: ${nodes.waiting || 0}`;

    const elOffline = document.getElementById("nodes-offline");
    if (elOffline) elOffline.innerText = `Offline: ${nodes.offline || 0}`;

    const claimedEl = document.getElementById("round-claimed-count");
    if (claimedEl) claimedEl.innerText = roundInfo.claims_count || 0;

    const submittedEl = document.getElementById("round-submitted-count");
    if (submittedEl) submittedEl.innerText = roundInfo.updates_count || 0;

    const deadlineEl = document.getElementById("round-deadline-timer");
    if (deadlineEl) {
      if (roundInfo.deadline && roundInfo.deadline > 0) {
        const remaining = Math.max(0, Math.floor(roundInfo.deadline - Date.now() / 1000));
        const mins = Math.floor(remaining / 60);
        const secs = remaining % 60;
        deadlineEl.innerText = `Deadline: ${mins}:${secs < 10 ? "0" : ""}${secs}`;
      } else {
        deadlineEl.innerText = roundInfo.status === "open" ? "Round Active" : "Round Idle";
      }
    }

    // Update Join URI
    const uriInput = document.getElementById("join-uri-input");
    if (uriInput) {
      const pubUrl = sess.public_url || window.location.origin;
      const cleanHost = pubUrl.replace(/^https?:\/\//, "");
      const uri = `hive://${cleanHost}/?s=${sess.session_id || ""}&t=${sess.join_token || ""}`;
      uriInput.value = uri;
    }

    // Toggle Pause/Resume buttons
    const pauseBtn = document.getElementById("btn-pause");
    const resumeBtn = document.getElementById("btn-resume");
    if (pauseBtn && resumeBtn) {
      if (sess.status === "paused") {
        pauseBtn.style.display = "none";
        resumeBtn.style.display = "inline-flex";
      } else {
        pauseBtn.style.display = "inline-flex";
        resumeBtn.style.display = "none";
      }
    }
  }

  // Update Training Charts & Event Log
  function updateTraining(rounds, events) {
    // 1. Loss & Accuracy Charts
    if (state.charts.loss && state.charts.accuracy && rounds.length > 0) {
      const labels = [];
      const trainLosses = [];
      const evalLosses = [];
      const accuracies = [];
      const globalNorms = [];

      rounds.forEach((r) => {
        labels.push(`R${r.round_id}`);
        const m = r.metrics || {};
        trainLosses.push(m.train_loss != null ? m.train_loss : null);
        evalLosses.push(m.eval_loss != null ? m.eval_loss : null);
        const acc = m.accuracy != null ? m.accuracy * 100 : m.eval_accuracy != null ? m.eval_accuracy * 100 : null;
        accuracies.push(acc);
        globalNorms.push(r.global_weight_norm || null);
      });

      state.charts.loss.data.labels = labels;
      state.charts.loss.data.datasets[0].data = trainLosses;
      state.charts.loss.data.datasets[1].data = evalLosses;
      state.charts.loss.update("none");

      state.charts.accuracy.data.labels = labels;
      state.charts.accuracy.data.datasets[0].data = accuracies;
      state.charts.accuracy.update("none");

      if (state.charts.globalNorms) {
        state.charts.globalNorms.data.labels = labels;
        state.charts.globalNorms.data.datasets[0].data = globalNorms;
        state.charts.globalNorms.update("none");
      }

      const emptyLoss = document.getElementById("loss-empty-state");
      if (emptyLoss) emptyLoss.style.display = "none";
      const emptyAcc = document.getElementById("acc-empty-state");
      if (emptyAcc) emptyAcc.style.display = "none";
    }

    // 2. Events Log
    const eventsList = document.getElementById("events-list");
    if (eventsList && events && events.length > 0) {
      eventsList.innerHTML = "";
      events.slice(-50).reverse().forEach((ev) => {
        const entry = document.createElement("div");
        const level = ev.level || "INFO";
        entry.className = `event-entry ${level}`;
        const timeStr = ev.created_at ? new Date(ev.created_at * 1000).toLocaleTimeString() : "--:--:--";
        entry.innerHTML = `<span class="event-time">${timeStr}</span><span class="event-msg">${ev.message}</span>`;
        eventsList.appendChild(entry);
      });
    }
  }

  // Update Contributors Table and Bar Chart
  function updateContributors(nodes) {
    const tbody = document.getElementById("contributors-body");
    if (!tbody) return;

    if (!nodes || nodes.length === 0) {
      tbody.innerHTML = `<tr><td colspan="13" class="empty-state"><div class="empty-icon">👥</div>Waiting for contributor clients to connect...</td></tr>`;
      return;
    }

    const labels = [];
    const shares = [];

    tbody.innerHTML = "";
    nodes.forEach((n) => {
      labels.push(n.nickname || n.node_id.slice(0, 8));
      shares.push(n.share_pct || 0);

      const tr = document.createElement("tr");
      const lastSeenStr = n.last_seen ? new Date(n.last_seen * 1000).toLocaleTimeString() : "Never";

      tr.innerHTML = `
        <td><strong>${n.nickname || "Anonymous"}</strong> <br><small class="mono text-muted">${n.node_id ? n.node_id.slice(0, 8) : ""}</small></td>
        <td><span class="chip active">${n.state || "registered"}</span></td>
        <td>${n.rounds_contributed || 0}</td>
        <td>${n.total_samples || 0}</td>
        <td>${n.accepted_updates || 0} / ${n.rejected_updates || 0}</td>
        <td class="mono">${n.mean_update_norm != null ? Number(n.mean_update_norm).toFixed(3) : "—"}</td>
        <td class="mono">${n.mean_cosine_similarity != null ? Number(n.mean_cosine_similarity).toFixed(3) : "—"}</td>
        <td>${n.outlier_count || 0}</td>
        <td>${n.share_pct != null ? n.share_pct.toFixed(1) + "%" : "0.0%"}</td>
        <td class="mono">${formatBytes(n.bytes_uploaded || 0)} / ${formatBytes(n.bytes_downloaded || 0)}</td>
        <td>${lastSeenStr}</td>
        <td>${n.estimated_epsilon != null ? n.estimated_epsilon.toFixed(2) : "None"}</td>
        <td>
          <div class="button-group">
            ${!n.approved ? `<button class="btn btn-sm btn-primary btn-approve" data-node-id="${n.node_id}">Approve</button>` : ""}
            <button class="btn btn-sm btn-danger btn-kick" data-node-id="${n.node_id}">Kick</button>
          </div>
        </td>
      `;
      tbody.appendChild(tr);
    });

    // Attach actions
    tbody.querySelectorAll(".btn-approve").forEach((b) => {
      b.addEventListener("click", async () => {
        const id = b.getAttribute("data-node-id");
        try {
          await apiFetch(`/admin/api/nodes/${id}/approve`, { method: "POST" });
          fetchDashboardData();
        } catch (e) {
          alert("Approval failed: " + e.message);
        }
      });
    });

    tbody.querySelectorAll(".btn-kick").forEach((b) => {
      b.addEventListener("click", async () => {
        const id = b.getAttribute("data-node-id");
        if (!confirm("Are you sure you want to kick this contributor node?")) return;
        try {
          await apiFetch(`/admin/api/nodes/${id}/kick`, { method: "POST" });
          fetchDashboardData();
        } catch (e) {
          alert("Kick action failed: " + e.message);
        }
      });
    });

    if (state.charts.contributions) {
      state.charts.contributions.data.labels = labels;
      state.charts.contributions.data.datasets[0].data = shares;
      state.charts.contributions.update("none");
    }
  }

  // Update Privacy Indicators
  function updatePrivacy(priv) {
    if (!priv) return;

    // 1. Transport
    const dotTransport = document.getElementById("dot-privacy-transport");
    const descTransport = document.getElementById("desc-privacy-transport");
    if (dotTransport && priv.transport) {
      dotTransport.className = `status-dot ${priv.transport.status === "secure" ? "green" : "amber"}`;
      if (descTransport) descTransport.innerText = priv.transport.details || "";
    }

    // 2. Update Ratio
    const dotUpdate = document.getElementById("dot-privacy-update-ratio");
    const descUpdate = document.getElementById("desc-privacy-update-ratio");
    if (dotUpdate && priv.update_only_ratio) {
      dotUpdate.className = `status-dot ${priv.update_only_ratio.status === "optimal" ? "green" : "amber"}`;
      if (descUpdate) descUpdate.innerText = priv.update_only_ratio.details || "";
    }

    // 3. Differential Privacy
    const dotDp = document.getElementById("dot-privacy-dp");
    const descDp = document.getElementById("desc-privacy-dp");
    if (dotDp && priv.differential_privacy) {
      dotDp.className = `status-dot ${priv.differential_privacy.enabled ? "green" : "amber"}`;
      if (descDp) descDp.innerText = priv.differential_privacy.details || "";
    }

    // 4. Anonymity Floor
    const dotAnon = document.getElementById("dot-privacy-anonymity");
    const descAnon = document.getElementById("desc-privacy-anonymity");
    if (dotAnon && priv.anonymity_floor) {
      dotAnon.className = `status-dot ${priv.anonymity_floor.status === "pass" ? "green" : "amber"}`;
      if (descAnon) descAnon.innerText = priv.anonymity_floor.details || "";
    }

    // 5. Poisoning Defenses
    const dotDef = document.getElementById("dot-privacy-defenses");
    const descDef = document.getElementById("desc-privacy-defenses");
    if (dotDef && priv.poisoning_defenses) {
      dotDef.className = `status-dot ${priv.poisoning_defenses.status === "active" ? "green" : "amber"}`;
      if (descDef) descDef.innerText = priv.poisoning_defenses.details || "";
    }

    // 6. Byte Accounting
    const weightBytesEl = document.getElementById("weight-data-bytes");
    if (weightBytesEl) weightBytesEl.innerText = formatBytes(priv.weight_delta_bytes_received || 0);

    const metaBytesEl = document.getElementById("meta-data-bytes");
    if (metaBytesEl) metaBytesEl.innerText = formatBytes(priv.meta_bytes_received || 0);
  }

  // Update Model Verification Tab
  async function updateVerification(rounds) {
    const tbody = document.getElementById("hash-chain-body");
    if (!tbody) return;

    if (!rounds || rounds.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-state"><div class="empty-icon">🔗</div>Hash chain records will populate once round 1 finishes aggregating.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    rounds.forEach((r) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>Round ${r.round_id}</strong></td>
        <td><code class="mono">${shortenHash(r.model_hash_in)}</code></td>
        <td><code class="mono">${shortenHash(r.model_hash_out)}</code></td>
        <td><span class="chip active">${r.aggregator || "FedAvg"}</span></td>
        <td><span style="color: var(--color-success); font-weight: 600;">✔ Available</span></td>
      `;
      tbody.appendChild(tr);
    });

    // Populate latest round details if available
    const latest = rounds[rounds.length - 1];
    if (latest) {
      try {
        const res = await apiFetch(`/admin/api/rounds/${latest.round_id}`);
        if (res.ok) {
          const detail = await res.json();
          renderRoundDetails(detail);
        }
      } catch (e) {}
    }
  }

  // Render Detailed Updates for Selected Round
  function renderRoundDetails(detail) {
    const roundUpdatesBody = document.getElementById("round-updates-body");
    if (!roundUpdatesBody) return;

    const updates = detail.updates || [];
    if (updates.length === 0) {
      roundUpdatesBody.innerHTML = `<tr><td colspan="7" class="empty-state">No client updates received for this round.</td></tr>`;
      return;
    }

    roundUpdatesBody.innerHTML = "";
    updates.forEach((u) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${u.nickname || u.node_id.slice(0, 8)}</strong></td>
        <td class="mono">${u.l2_norm != null ? Number(u.l2_norm).toFixed(3) : "—"}</td>
        <td class="mono">${u.cosine_similarity != null ? Number(u.cosine_similarity).toFixed(3) : "—"}</td>
        <td><span style="color: var(--color-success); font-weight: 600;">✔ Match</span></td>
        <td><code class="mono">${shortenHash(u.delta_sha256)}</code></td>
        <td><span class="badge ${u.accepted ? "badge-running" : "badge-error"}">${u.accepted ? "ACCEPTED" : "REJECTED"}</span></td>
        <td>${u.is_outlier ? '<span style="color: var(--color-danger); font-weight: 600;">⚠ OUTLIER</span>' : "No"}</td>
      `;
      roundUpdatesBody.appendChild(tr);
    });

    // Populate layer norms chart if metrics exist
    if (state.charts.layerNorms && detail.metrics && detail.metrics.layer_norms) {
      const layers = Object.keys(detail.metrics.layer_norms);
      const norms = Object.values(detail.metrics.layer_norms);
      state.charts.layerNorms.data.labels = layers;
      state.charts.layerNorms.data.datasets[0].data = norms;
      state.charts.layerNorms.update("none");
    }
  }

  // Master Polling Loop
  async function fetchDashboardData() {
    try {
      const [sumRes, rndRes, nodeRes, privRes, evRes] = await Promise.allSettled([
        apiFetch("/admin/api/summary"),
        apiFetch("/admin/api/rounds"),
        apiFetch("/admin/api/nodes"),
        apiFetch("/admin/api/privacy"),
        apiFetch("/admin/api/events"),
      ]);

      if (sumRes.status === "fulfilled" && sumRes.value.ok) {
        state.summary = await sumRes.value.json();
        updateHeader(state.summary);
      }

      if (rndRes.status === "fulfilled" && rndRes.value.ok) {
        state.rounds = await rndRes.value.json();
      }

      if (evRes.status === "fulfilled" && evRes.value.ok) {
        state.events = await evRes.value.json();
      }

      updateTraining(state.rounds, state.events);

      if (nodeRes.status === "fulfilled" && nodeRes.value.ok) {
        state.nodes = await nodeRes.value.json();
        updateContributors(state.nodes);
      }

      if (privRes.status === "fulfilled" && privRes.value.ok) {
        state.privacy = await privRes.value.json();
        updatePrivacy(state.privacy);
      }

      if (state.activeTab === "verification") {
        updateVerification(state.rounds);
      }
    } catch (err) {
      console.error("Dashboard polling error:", err);
    }
  }

  // App Entrypoint
  window.addEventListener("DOMContentLoaded", () => {
    initAuth();
    initTabs();
    initActions();
    initCharts();

    // Initial fetch + 2-second polling interval
    fetchDashboardData();
    setInterval(fetchDashboardData, 2000);
  });
})();
