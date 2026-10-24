# Hive: Real-World Architecture & Production Deployment Guide

This guide explains how **Hive** works in real-world environments. It covers how a **Model Author (Host)** prepares their own custom PyTorch model, opens a public training session, and distributes a cryptographic invitation link, as well as how **Contributors** join the session, train the model locally without exposing raw data, and stream weight deltas back to the Host.

---

## 1. Host Workflow: Preparing Your Model & Opening a Session

### Step 1: Define Your Custom Model & Preprocessor
Organize your model repository into a package directory (e.g., `my_model_pkg/`):

```
my_model_pkg/
├── manifest.json       # Hyperparameters, metrics, and entrypoints
├── model.py            # Standard PyTorch nn.Module architecture
├── preprocess.py       # Local dataset validator & PyTorch Dataset builder
└── requirements.txt    # Python runtime dependencies
```

#### `model.py` (Your Custom Model)
```python
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
```

#### `preprocess.py` (Local Data Validator & DataLoader)
Contributors never send data to the server. The preprocessor defines how their local files are validated and converted into a PyTorch `Dataset`:

```python
from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import TensorDataset
from hive.core.contracts import BaseDataPreprocessor, ValidationResult

class MyPreprocessor(BaseDataPreprocessor):
    def validate_local_directory(self, data_dir: Path | str) -> ValidationResult:
        p = Path(data_dir)
        data_file = p / "dataset.pt"
        if not data_file.exists():
            return ValidationResult(ok=False, num_samples=0, message="dataset.pt missing")
        samples = len(torch.load(data_file)["x"])
        return ValidationResult(ok=True, num_samples=samples, message=f"Found {samples} samples")

    def build_dataset(self, data_dir: Path | str, split: Literal["train", "eval"], seed: int = 42) -> TensorDataset:
        d = torch.load(Path(data_dir) / "dataset.pt")
        x, y = d["x"], d["y"]
        # Split train / eval
        split_idx = int(len(x) * 0.8)
        return TensorDataset(x[:split_idx], y[:split_idx]) if split == "train" else TensorDataset(x[split_idx:], y[split_idx:])
```

#### `manifest.json` (Specification)
```json
{
  "package_name": "cancer-detection-net",
  "version": "1.0.0",
  "description": "Collaborative training for medical diagnostic classifier",
  "model_entrypoint": "model.py:MyModel",
  "preprocessor_entrypoint": "preprocess.py:MyPreprocessor",
  "loss": "cross_entropy",
  "metrics": "accuracy",
  "hyperparameters": {
    "local_epochs": 3,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "adam"
  },
  "data_requirements": {
    "description": "Folder with dataset.pt containing 128-dim feature tensors",
    "min_samples": 50
  }
}
```

---

### Step 2: Build & Sign the `.hivepkg` Archive
Generate an Ed25519 signing key (if not already created) and compile the signed package:

```bash
# 1. Generate author keypair
hive keygen -o my_host_key.pem

# 2. Build and sign package
hive package build \
  --src my_model_pkg \
  --key my_host_key.pem \
  --out model.hivepkg
```

---

### Step 3: Launch the Host Server on Public Network

Hosts can run directly on a VPS, a cloud VM, or a local machine using public tunneling.

#### Mode A: Cloud / Public IP / Domain
```bash
hive host \
  --package model.hivepkg \
  --public-url "https://fl.yourdomain.com" \
  --tunnel manual \
  --rounds 10 \
  --min-clients 5 \
  --aggregation fedavg
```

#### Mode B: Local Host with Free Automatic Cloudflare Tunneling
If you do not have a public static IP or open firewall ports:
```bash
hive host \
  --package model.hivepkg \
  --tunnel cloudflared \
  --rounds 10 \
  --min-clients 3
```

---

### Step 4: Retrieve the Invitation Link
When the host boots, it outputs the invitation link:

```
======================================================================
  HIVE FEDERATED LEARNING HOST SERVER STARTED
======================================================================
Session ID:         sess_79f7d228f587
Package Name:       cancer-detection-net v1.0.0
Author Fingerprint: ed25519:b11a732343c0a989
Transport:          https://fl.yourdomain.com (HTTPS)
----------------------------------------------------------------------
 JOIN INVITATION LINK (Share with contributors):
   hive://fl.yourdomain.com/sess_79f7d228f587?t=cbqB85l3yQ3eqHcO&fp=ed25519%3Ab11a732343c0a989&tls=1
----------------------------------------------------------------------
 HOST ADMIN DASHBOARD:
   http://127.0.0.1:8000/?token=Xz0vOaeULfJsvuhc5YA2eQ
======================================================================
```

---

## 3. Contributor Workflow: Joining & Training Without Data Leakage

Contributors do not need cloud servers. They install Hive on their workstation, laptop, or server.

### Step 1: Contributor Joins via Invitation Link
The contributor runs one command with the link provided by the Host:

```bash
hive join \
  --uri "hive://fl.yourdomain.com/sess_79f7d228f587?t=cbqB85l3yQ3eqHcO&fp=ed25519%3Ab11a732343c0a989&tls=1" \
  --data-dir "/home/user/private_medical_data" \
  --nickname "Hospital-North"
```

### Step 2: What Happens on the Contributor's Machine
Hive automatically launches a local browser interface at `http://127.0.0.1:8765/?token=...`:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CONTRIBUTOR LOCAL DASHBOARD                     │
├─────────────┬─────────────────┬─────────────┬────────────┬─────────────┤
│ 1. CONNECT  │ 2. CODE AUDIT   │ 3. DATA     │ 4. CONTROLS│ 5. TRAINING │
└─────────────┴─────────────────┴─────────────┴────────────┴─────────────┘
```

1. **Cryptographic Verification**:
   The client downloads the `.hivepkg` package and matches the author's public key fingerprint against the `fp=` parameter in the URI. If an attacker replaced the model, it is instantly rejected.
2. **Automated Security Scan (AST Lint)**:
   The client's local parser checks the Python AST. Forbidden syscalls (socket connections, file exfiltration, subprocessing) trigger a security warning.
3. **Data Inspection in Sandbox**:
   The client runs `preprocess.py` inside an isolated subprocess or container. Raw patient/private data stays on the local disk. Only the sample count (e.g., 250 samples) is reported to the Host.
4. **Hardware Governor Settings**:
   Contributors set constraints:
   - Cap CPU usage (e.g., max 40% CPU).
   - Only train when system is idle for 5 minutes.
   - Pause when on laptop battery power.
   - Schedule training hours (e.g., 10 PM to 6 AM).

---

## 4. End-to-End Distributed Round Lifecycle

Below is the complete network protocol executed during each federated round:

```
CONTRIBUTOR 1                      HOST SERVER                      CONTRIBUTOR 2
      |                                 |                                 |
      |--------- POST /v1/join -------->|                                 |
      |                                 |<-------- POST /v1/join ---------|
      |--------- POST /v1/ready ------->|                                 |
      |                                 |<-------- POST /v1/ready --------|
      |                                 |                                 |
      |                        [Check min_clients >= 2]                   |
      |                        [OPEN ROUND #1]                            |
      |                                 |                                 |
      |<=== GET /v1/round/1/model ======|====== GET /v1/round/1/model ===>|
      |     (Downloads safetensors)     |     (Downloads safetensors)     |
      |                                 |                                 |
      | [Trains on private data]        |        [Trains on private data] |
      | [Computes ΔW1 = W1 - W_glob]    |        [Computes ΔW2 = W2-W_glob|
      |                                 |                                 |
      |--------- POST /v1/round/1/update------->|                         |
      |          (Uploads only ΔW1)     |                                 |
      |                                 |<------- POST /v1/round/1/update-|
      |                                 |         (Uploads only ΔW2)      |
      |                                 |                                 |
      |                         [AGGREGATION (FedAvg)]                    |
      |                         W_new = W_glob + (ΔW1 + ΔW2)/2            |
      |                         [SAVE CHECKPOINT: round_1.safetensors]    |
      |                         [ADVANCE TO ROUND #2]                     |
      |                                 |                                 |
      |<=== GET /v1/round/2/model ======|====== GET /v1/round/2/model ===>|
```

### Privacy Guarantee: "What Leaves My Device"
In the Contributor Dashboard, clicking the **What Leaves My Device** transparency tab reveals the exact transmission payload:
- Tensor names: `["encoder.0.weight", "encoder.0.bias", "encoder.2.weight", ...]`
- Sizes & Data types: `float32`, 504 bytes total.
- Cryptographic hash: `sha256:d8f7e67f99dcff80...`
- **Zero raw training records, zero image files, and zero text entries leave the machine.**

---

## 5. Session Termination & Connection Lifecycle

### Scenario A: Contributor Leaves (Voluntary)
- Contributor clicks **Leave Session** in their UI or issues `Ctrl+C` in their terminal.
- Contributor sends `POST /v1/leave` with their authentication credentials.
- The host marks the node as `left` and re-normalizes the aggregation weights across remaining nodes.
- Local temporary sandbox scratch spaces on the contributor machine are purged.

### Scenario B: Contributor Drops Offline or Encounters Timeout
- If a contributor loses internet connectivity mid-round, the Host waits until `round_timeout_s` expires.
- If at least `min_clients` submitted updates, the Host completes the round with the received updates.
- If too few updates arrived, the round is aborted and rescheduled without penalty to compliant nodes.

### Scenario C: Training Session Completed (Host Finalization)
1. Once `total_rounds` are executed, the Host Orchestrator sets status to `completed`.
2. The server outputs `final_model.safetensors` in the host working directory.
3. The Host Admin audits the model integrity:
   ```bash
   curl -X POST "http://127.0.0.1:8000/model/verify?token=<ADMIN_TOKEN>"
   ```
4. Output confirms cryptographic continuity:
   ```json
   {
     "verified": true,
     "target_file": "final_model.safetensors",
     "match": true,
     "tampered": false
   }
   ```
5. All connected contributor nodes acknowledge the session completion and return to standby.

---

## 6. Real-World Security & Defense Summary

| Threat | Hive Mitigation |
|---|---|
| **Malicious Package Injection** | Packages are cryptographically signed with Ed25519. The Join URI pins the author key fingerprint. Contributor clients refuse any mismatched package. |
| **Exploitative Python Code** | Contributor agents parse the package AST before execution, rejecting socket connections, shell execution, or unauthorized filesystem access. |
| **Data Exfiltration** | Training runs inside isolated process/Docker sandboxes with restricted network egress. Only parameter weight matrices ($\Delta W$) are transmitted. |
| **Model Inversion / Gradient Leakage** | Optional Differential Privacy (DP) clips gradient norms and injects Gaussian noise ($\epsilon, \delta$) to protect training data distributions. |
| **Byzantine Poisoning Attacks** | Hosts can select robust aggregation strategies (`trimmed-mean`, `median`, `krum`) to filter out poisoned client weights. |
