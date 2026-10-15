/**
 * Hive Contributor Client Local Web UI
 * Zero-dependency vanilla JS, stepper navigation, live resource toggles,
 * code viewer, folder picker modal, and payload transparency.
 */

(function () {
  "use strict";

  const state = {
    token: null,
    currentStep: 1,
    maxUnlockedStep: 1,
    status: null,
    resources: null,
    packageReview: null,
    history: [],
    payloadPreview: null,
    currentBrowsePath: "",
    pollTimer: null,
  };

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
  }

  function initAuth() {
    const params = new URLSearchParams(window.location.search);
    const queryToken = params.get("token");
    if (queryToken) {
      state.token = queryToken;
      try {
        sessionStorage.setItem("hive_ui_token", queryToken);
      } catch (e) {}
    } else {
      try {
        state.token = sessionStorage.getItem("hive_ui_token");
      } catch (e) {}
    }
  }

  async function apiFetch(url, options = {}) {
    const headers = options.headers || {};
    if (state.token) {
      headers["X-Client-Token"] = state.token;
    }
    options.headers = headers;
    options.credentials = "same-origin";

    const res = await fetch(url, options);
    if (!res.ok && res.status === 401) {
      console.warn("UI token authentication failed");
    }
    return res;
  }

  // Stepper Transition
  function goToStep(step) {
    if (step > state.maxUnlockedStep) return;

    state.currentStep = step;
    document.querySelectorAll(".step-item").forEach((el) => {
      const s = parseInt(el.getAttribute("data-step"), 10);
      el.classList.remove("active");
      if (s === step) el.classList.add("active");
      if (s < step) el.classList.add("completed");
    });

    document.querySelectorAll(".screen-card").forEach((card) => {
      card.classList.remove("active");
    });

    const screens = ["connect", "review", "data", "controls", "status"];
    const activeScreen = document.getElementById(`screen-${screens[step - 1]}`);
    if (activeScreen) {
      activeScreen.classList.add("active");
    }

    if (step === 2) loadPackageReview();
    if (step === 4) loadResourceControls();
  }

  function unlockStep(step) {
    if (step > state.maxUnlockedStep) {
      state.maxUnlockedStep = step;
    }
    goToStep(step);
  }

  // Setup Stepper Nav Click Handlers
  function initStepper() {
    document.querySelectorAll(".step-item").forEach((el) => {
      el.addEventListener("click", () => {
        const step = parseInt(el.getAttribute("data-step"), 10);
        if (step <= state.maxUnlockedStep) {
          goToStep(step);
        }
      });
    });

    document.querySelectorAll(".btn-prev-step").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = parseInt(btn.getAttribute("data-target"), 10);
        goToStep(target);
      });
    });
  }

  // Screen 1: Connect Flow
  function initConnectScreen() {
    const btn = document.getElementById("btn-connect");
    const uriInput = document.getElementById("input-uri");
    const nickInput = document.getElementById("input-nickname");
    const errBox = document.getElementById("connect-error");

    // Populate URI from URL hash or query if provided
    const params = new URLSearchParams(window.location.search);
    if (params.get("uri")) {
      uriInput.value = params.get("uri");
    }

    btn.addEventListener("click", async () => {
      errBox.classList.remove("show");
      errBox.innerText = "";
      const uri = uriInput.value.trim();
      const nickname = nickInput.value.trim() || "Contributor";

      if (!uri) {
        errBox.innerText = "Please provide a valid hive:// session URI.";
        errBox.classList.add("show");
        return;
      }

      btn.disabled = true;
      btn.innerText = "Connecting & Downloading Package...";

      try {
        const res = await apiFetch("/local/connect", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ uri, nickname }),
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail?.message || data.detail || "Connection failed");
        }

        unlockStep(2);
      } catch (err) {
        errBox.innerText = err.message || "Failed to connect to host session";
        errBox.classList.add("show");
      } finally {
        btn.disabled = false;
        btn.innerText = "Connect & Download Package ➔";
      }
    });
  }

  // Screen 2: Review & Approve Flow
  async function loadPackageReview() {
    try {
      const res = await apiFetch("/local/package-review");
      if (!res.ok) return;
      const data = await res.json();
      state.packageReview = data;

      // Fingerprint grouping (e.g., 4-character chunks)
      const fp = data.pinned_fingerprint || "UNPINNED";
      const fpEl = document.getElementById("review-fingerprint");
      if (fpEl) {
        fpEl.innerText = fp.match(/.{1,4}/g)?.join(":") || fp;
      }

      // Manifest & Requirements
      const descEl = document.getElementById("review-description");
      if (descEl) descEl.innerText = data.manifest?.description || "No description provided.";

      const reqEl = document.getElementById("review-requirements");
      if (reqEl) reqEl.innerText = data.data_requirements?.description || "Any supported format";

      const digestEl = document.getElementById("review-digest");
      if (digestEl) digestEl.innerText = data.content_digest || "—";

      // File List
      const fileListEl = document.getElementById("review-file-list");
      if (fileListEl && data.sources) {
        fileListEl.innerHTML = "";
        Object.keys(data.sources).forEach((fname) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.innerText = fname;
          fileListEl.appendChild(chip);
        });
      }

      // Setup Code Display
      const codeDisplay = document.getElementById("code-display");
      const sources = data.sources || {};
      if (sources["model.py"] && codeDisplay) {
        codeDisplay.innerText = sources["model.py"];
      }

      // Setup Tabs
      const tabs = document.querySelectorAll(".code-tab-btn");
      tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
          tabs.forEach((t) => t.classList.remove("active"));
          tab.classList.add("active");
          const fname = tab.getAttribute("data-file");
          if (codeDisplay && sources[fname]) {
            codeDisplay.innerText = sources[fname];
          }
        });
      });

      // Lint warnings
      const lintList = document.getElementById("review-lint-list");
      if (lintList) {
        const findings = data.lint_findings || [];
        if (findings.length === 0) {
          lintList.innerHTML = `<span style="color: var(--color-success); font-weight: 600;">✔ No security warnings detected by static AST analyzer.</span>`;
        } else {
          lintList.innerHTML = "";
          findings.forEach((f) => {
            const item = document.createElement("div");
            item.style.padding = "0.25rem 0";
            item.innerHTML = `<span style="color: var(--color-warning); font-weight: 700;">[${f.severity || "WARN"}]</span> ${f.file}:${f.line} — ${f.message}`;
            lintList.appendChild(item);
          });
        }
      }

      // Sandbox Mode & Consent Button
      const sandboxMode = state.status?.sandbox_mode || "process";
      const modeBadge = document.getElementById("review-sandbox-mode");
      const ackBox = document.getElementById("check-sandbox-ack");
      const approveBtn = document.getElementById("btn-approve-package");

      if (sandboxMode === "docker") {
        if (modeBadge) {
          modeBadge.innerText = "Docker Container (Isolated)";
          modeBadge.className = "badge badge-success";
        }
        if (ackBox) {
          ackBox.checked = true;
          document.getElementById("sandbox-ack-container").style.display = "none";
        }
        if (approveBtn) approveBtn.disabled = false;
      } else {
        if (modeBadge) {
          modeBadge.innerText = "Process Mode (Reduced Isolation)";
          modeBadge.className = "badge badge-warning";
        }
        if (ackBox && approveBtn) {
          approveBtn.disabled = !ackBox.checked;
          ackBox.addEventListener("change", () => {
            approveBtn.disabled = !ackBox.checked;
          });
        }
      }
    } catch (e) {
      console.error("Error loading package review:", e);
    }
  }

  function initReviewScreen() {
    const approveBtn = document.getElementById("btn-approve-package");
    const declineBtn = document.getElementById("btn-decline-package");

    approveBtn.addEventListener("click", async () => {
      try {
        const digest = state.packageReview?.content_digest;
        const res = await apiFetch("/local/consent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ digest }),
        });
        if (res.ok) {
          unlockStep(3);
        }
      } catch (e) {
        alert("Failed to record consent: " + e.message);
      }
    });

    declineBtn.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to decline? Local cache will be deleted and you will disconnect.")) return;
      try {
        await apiFetch("/local/leave", { method: "POST" });
        location.reload();
      } catch (e) {
        location.reload();
      }
    });
  }

  // Screen 3: Data Folder Selection & Browser Modal
  function initDataScreen() {
    const browseBtn = document.getElementById("btn-browse-folder");
    const validateBtn = document.getElementById("btn-validate-data");
    const continueBtn = document.getElementById("btn-data-continue");
    const pathInput = document.getElementById("input-data-path");
    const resultBox = document.getElementById("data-validation-result");
    const sampleBadge = document.getElementById("sample-count-badge");

    // Modal elements
    const modal = document.getElementById("folder-modal");
    const closeModal = document.getElementById("btn-close-modal");
    const cancelModal = document.getElementById("btn-modal-cancel");
    const selectModal = document.getElementById("btn-modal-select");
    const curPathEl = document.getElementById("modal-current-path");
    const dirListEl = document.getElementById("modal-dir-list");

    async function loadDirectory(path = "") {
      try {
        const res = await apiFetch(`/local/fs?path=${encodeURIComponent(path)}`);
        if (!res.ok) return;
        const data = await res.json();
        state.currentBrowsePath = data.current;
        curPathEl.innerText = data.current;
        dirListEl.innerHTML = "";

        if (data.parent) {
          const parentItem = document.createElement("div");
          parentItem.className = "dir-item";
          parentItem.innerHTML = `<span>📁</span> <strong>.. (Parent Directory)</strong>`;
          parentItem.addEventListener("click", () => loadDirectory(data.parent));
          dirListEl.appendChild(parentItem);
        }

        const dirs = data.directories || [];
        if (dirs.length === 0) {
          const empty = document.createElement("div");
          empty.style.padding = "0.75rem";
          empty.style.color = "var(--text-muted)";
          empty.innerText = "No subdirectories found in this folder.";
          dirListEl.appendChild(empty);
        } else {
          dirs.forEach((d) => {
            const item = document.createElement("div");
            item.className = "dir-item";
            item.innerHTML = `<span>📁</span> <span>${d.name}</span>`;
            item.addEventListener("click", () => loadDirectory(d.path));
            dirListEl.appendChild(item);
          });
        }
      } catch (e) {
        console.error("FS browse error:", e);
      }
    }

    browseBtn.addEventListener("click", () => {
      modal.classList.add("show");
      loadDirectory(pathInput.value.trim());
    });

    closeModal.addEventListener("click", () => modal.classList.remove("show"));
    cancelModal.addEventListener("click", () => modal.classList.remove("show"));
    selectModal.addEventListener("click", () => {
      pathInput.value = state.currentBrowsePath;
      modal.classList.remove("show");
    });

    validateBtn.addEventListener("click", async () => {
      const p = pathInput.value.trim();
      if (!p) {
        alert("Please enter or select a valid local folder path.");
        return;
      }

      validateBtn.disabled = true;
      validateBtn.innerText = "Validating in Sandbox...";
      resultBox.style.display = "none";

      try {
        const res = await apiFetch("/local/data-dir", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: p }),
        });

        const data = await res.json();
        if (res.ok && data.ok) {
          resultBox.style.display = "flex";
          sampleBadge.className = "badge badge-success";
          sampleBadge.innerText = `✔ ${data.num_samples} Samples Verified`;
          continueBtn.disabled = false;
        } else {
          resultBox.style.display = "flex";
          sampleBadge.className = "badge badge-danger";
          sampleBadge.innerText = `❌ Validation Failed`;
          alert("Validation error: " + (data.message || data.detail || "Requirements not met"));
        }
      } catch (err) {
        alert("Data validation call failed: " + err.message);
      } finally {
        validateBtn.disabled = false;
        validateBtn.innerText = "Validate Dataset";
      }
    });

    continueBtn.addEventListener("click", () => {
      unlockStep(4);
    });
  }

  // Screen 4: Controls Flow & Live Resource Sync
  async function loadResourceControls() {
    try {
      const res = await apiFetch("/local/resources");
      if (!res.ok) return;
      const pol = await res.json();
      state.resources = pol;

      const idleToggle = document.getElementById("toggle-only-idle");
      const cpuSlider = document.getElementById("slider-max-cpu");
      const threshSlider = document.getElementById("slider-cpu-threshold");
      const delaySlider = document.getElementById("slider-idle-minutes");
      const roundsInput = document.getElementById("input-max-rounds");
      const ramInput = document.getElementById("input-max-ram");
      const gpuToggle = document.getElementById("toggle-use-gpu");
      const battToggle = document.getElementById("toggle-pause-battery");

      if (idleToggle) idleToggle.checked = !!pol.only_when_idle;
      if (cpuSlider) {
        cpuSlider.value = pol.max_cpu_percent || 50;
        document.getElementById("val-max-cpu").innerText = `${cpuSlider.value}%`;
      }
      if (threshSlider) {
        threshSlider.value = pol.idle_cpu_threshold || 50;
        document.getElementById("val-cpu-threshold").innerText = `${threshSlider.value}%`;
      }
      if (delaySlider) {
        const mins = Math.round((pol.idle_time_seconds || 180) / 60);
        delaySlider.value = mins;
        document.getElementById("val-idle-minutes").innerText = `${mins} min`;
      }
      if (roundsInput) roundsInput.value = pol.max_rounds || 0;
      if (ramInput) ramInput.value = pol.max_ram_mb || 4096;
      if (gpuToggle) gpuToggle.checked = !!pol.use_gpu;
      if (battToggle) battToggle.checked = !!pol.pause_on_battery;
    } catch (e) {
      console.error("Failed to load resource policy:", e);
    }
  }

  function initControlsScreen() {
    const idleToggle = document.getElementById("toggle-only-idle");
    const cpuSlider = document.getElementById("slider-max-cpu");
    const threshSlider = document.getElementById("slider-cpu-threshold");
    const delaySlider = document.getElementById("slider-idle-minutes");
    const roundsInput = document.getElementById("input-max-rounds");
    const ramInput = document.getElementById("input-max-ram");
    const gpuToggle = document.getElementById("toggle-use-gpu");
    const battToggle = document.getElementById("toggle-pause-battery");

    const pauseBtn = document.getElementById("btn-pause-client");
    const leaveBtn = document.getElementById("btn-leave-client");
    const continueBtn = document.getElementById("btn-controls-continue");

    async function pushResourceUpdate() {
      const payload = {
        only_when_idle: idleToggle?.checked ?? true,
        max_cpu_percent: parseInt(cpuSlider?.value || 50, 10),
        idle_cpu_threshold: parseInt(threshSlider?.value || 50, 10),
        idle_time_seconds: parseInt(delaySlider?.value || 3, 10) * 60,
        max_rounds: parseInt(roundsInput?.value || 0, 10),
        max_ram_mb: parseInt(ramInput?.value || 4096, 10),
        use_gpu: gpuToggle?.checked ?? false,
        pause_on_battery: battToggle?.checked ?? false,
      };

      try {
        await apiFetch("/local/resources", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      } catch (e) {
        console.error("Resource PUT error:", e);
      }
    }

    [idleToggle, gpuToggle, battToggle].forEach((t) => {
      if (t) t.addEventListener("change", pushResourceUpdate);
    });

    if (cpuSlider) {
      cpuSlider.addEventListener("input", () => {
        document.getElementById("val-max-cpu").innerText = `${cpuSlider.value}%`;
      });
      cpuSlider.addEventListener("change", pushResourceUpdate);
    }

    if (threshSlider) {
      threshSlider.addEventListener("input", () => {
        document.getElementById("val-cpu-threshold").innerText = `${threshSlider.value}%`;
      });
      threshSlider.addEventListener("change", pushResourceUpdate);
    }

    if (delaySlider) {
      delaySlider.addEventListener("input", () => {
        document.getElementById("val-idle-minutes").innerText = `${delaySlider.value} min`;
      });
      delaySlider.addEventListener("change", pushResourceUpdate);
    }

    [roundsInput, ramInput].forEach((inp) => {
      if (inp) inp.addEventListener("change", pushResourceUpdate);
    });

    pauseBtn.addEventListener("click", async () => {
      const isPaused = state.status?.is_user_paused;
      const endpoint = isPaused ? "/local/resume" : "/local/pause";
      try {
        await apiFetch(endpoint, { method: "POST" });
        pauseBtn.innerText = isPaused ? "⏸ Pause Training" : "▶ Resume Training";
      } catch (e) {
        alert("Action failed: " + e.message);
      }
    });

    leaveBtn.addEventListener("click", async () => {
      if (!confirm("Leave federated session and delete local job cache?")) return;
      try {
        await apiFetch("/local/leave", { method: "POST" });
        location.reload();
      } catch (e) {
        location.reload();
      }
    });

    continueBtn.addEventListener("click", () => {
      unlockStep(5);
    });
  }

  // Screen 5: Status, History & Transparency
  function updateStatusUI(status, history, preview) {
    if (!status) return;

    // Badges & Gating
    const stateBadge = document.getElementById("client-state-badge");
    if (stateBadge) {
      stateBadge.innerText = status.state || "UNKNOWN";
      stateBadge.className = "badge";
      if (status.state === "READY") stateBadge.classList.add("badge-success");
      else if (status.state === "TRAINING") stateBadge.classList.add("badge-info");
      else if (status.state === "PAUSED_LOCAL") stateBadge.classList.add("badge-warning");
      else if (status.state === "ERROR") stateBadge.classList.add("badge-danger");
      else stateBadge.classList.add("badge-info");
    }

    const stateReason = document.getElementById("client-state-reason");
    if (stateReason) stateReason.innerText = status.state_reason || "Ready for rounds.";

    const roundDisplay = document.getElementById("client-round-display");
    if (roundDisplay) {
      const cur = status.session_info?.current_round || 0;
      const tot = status.session_info?.total_rounds || 0;
      roundDisplay.innerText = `Round ${cur} / ${tot}`;
    }

    const bytesEl = document.getElementById("client-bytes-uploaded");
    if (bytesEl) bytesEl.innerText = formatBytes(status.bytes_uploaded || 0);

    const gateNote = document.getElementById("gating-note-text");
    if (gateNote) {
      gateNote.innerText = status.gating_reason || (status.can_train ? "Ready: all resource conditions met." : "Waiting on resource gating...");
    }

    // Live training note
    const liveNote = document.getElementById("live-training-progress");
    if (liveNote) {
      if (status.state === "TRAINING") {
        liveNote.innerText = `⚡ Sandbox Active: Training local PyTorch model on verified dataset...`;
        liveNote.style.color = "var(--color-primary)";
      } else if (status.state === "UPLOADING") {
        liveNote.innerText = `⬆ Submitting encrypted weight update delta to Host server...`;
        liveNote.style.color = "var(--color-accent)";
      } else {
        liveNote.innerText = `Idle — Ready for next federated training round from Host.`;
        liveNote.style.color = "var(--text-muted)";
      }
    }

    // Update History Table
    const histBody = document.getElementById("client-history-body");
    if (histBody && history && history.length > 0) {
      histBody.innerHTML = "";
      history.forEach((h) => {
        const tr = document.createElement("tr");
        const lossStr = h.train_loss != null ? Number(h.train_loss).toFixed(4) : "—";
        const accStr = h.eval_accuracy != null ? (Number(h.eval_accuracy) * 100).toFixed(1) + "%" : "—";
        const durStr = h.duration_s != null ? `${Number(h.duration_s).toFixed(1)}s` : "—";
        const kbStr = h.bytes_uploaded != null ? formatBytes(h.bytes_uploaded) : "—";

        tr.innerHTML = `
          <td><strong>Round ${h.round_id}</strong></td>
          <td class="mono">${lossStr}</td>
          <td class="mono">${accStr}</td>
          <td>${durStr}</td>
          <td class="mono">${kbStr}</td>
          <td><span class="badge ${h.accepted !== false ? "badge-success" : "badge-danger"}">${h.accepted !== false ? "YES" : "REJECTED"}</span></td>
        `;
        histBody.appendChild(tr);
      });
    }

    // Update Payload Transparency Panel
    if (preview && preview.preview) {
      const p = preview.preview;
      const countEl = document.getElementById("preview-tensor-count");
      if (countEl) countEl.innerText = `${p.tensor_count || 0} weight tensors`;

      const pBytesEl = document.getElementById("preview-bytes");
      if (pBytesEl) pBytesEl.innerText = formatBytes(p.bytes || 0);

      const pShaEl = document.getElementById("preview-sha256");
      if (pShaEl) pShaEl.innerText = p.sha256 || "—";

      const listEl = document.getElementById("preview-tensors-list");
      if (listEl && p.tensors && p.tensors.length > 0) {
        listEl.innerHTML = "";
        p.tensors.forEach((t) => {
          const row = document.createElement("div");
          row.style.padding = "0.125rem 0";
          row.innerText = `${t.name}: shape [${t.shape?.join(", ")}], dtype=${t.dtype}`;
          listEl.appendChild(row);
        });
      }
    }

    // Footer Updates
    const fNode = document.getElementById("footer-node-id");
    if (fNode) fNode.innerText = `Node: ${status.node_id ? status.node_id.slice(0, 10) : "Not Joined"} (${status.nickname || "Anon"})`;

    const fSandbox = document.getElementById("footer-sandbox");
    if (fSandbox) {
      fSandbox.innerText = status.sandbox_mode === "docker" ? "Docker Sandbox (Isolated)" : "Process Mode";
      fSandbox.className = `badge ${status.sandbox_mode === "docker" ? "badge-success" : "badge-warning"}`;
    }

    const fGate = document.getElementById("footer-gating-note");
    if (fGate) fGate.innerText = status.can_train ? "Active" : "Throttled / Gated";

    const fBytes = document.getElementById("footer-bytes-uploaded");
    if (fBytes) fBytes.innerText = `Uploaded: ${formatBytes(status.bytes_uploaded || 0)}`;

    // Auto-advance stepper if state progresses
    if (status.state === "AWAITING_CONSENT" && state.maxUnlockedStep < 2) unlockStep(2);
    if ((status.state === "AWAITING_DATA" || status.state === "VALIDATING_DATA") && state.maxUnlockedStep < 3) unlockStep(3);
    if ((status.state === "READY" || status.state === "TRAINING" || status.state === "UPLOADING") && state.maxUnlockedStep < 5) unlockStep(5);
  }

  // Master Polling Loop
  async function pollStatus() {
    try {
      const [stRes, histRes, prevRes] = await Promise.allSettled([
        apiFetch("/local/status"),
        apiFetch("/local/history"),
        apiFetch("/local/payload-preview"),
      ]);

      if (stRes.status === "fulfilled" && stRes.value.ok) {
        state.status = await stRes.value.json();
      }

      if (histRes.status === "fulfilled" && histRes.value.ok) {
        state.history = await histRes.value.json();
      }

      if (prevRes.status === "fulfilled" && prevRes.value.ok) {
        state.payloadPreview = await prevRes.value.json();
      }

      updateStatusUI(state.status, state.history, state.payloadPreview);
    } catch (err) {
      console.error("Local polling error:", err);
    }
  }

  // App Initialization
  window.addEventListener("DOMContentLoaded", () => {
    initAuth();
    initStepper();
    initConnectScreen();
    initReviewScreen();
    initDataScreen();
    initControlsScreen();

    // Initial status poll & 2s interval
    pollStatus();
    state.pollTimer = setInterval(pollStatus, 2000);
  });
})();
