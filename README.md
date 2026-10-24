# Hive

**Train smarter AI models together — without sharing your data.**

---

Hive is an open-source tool that lets people collaborate to train machine learning models, even when their data is private and cannot be shared. Think of it like a community science project: many volunteers contribute computing time and local expertise, but their raw recordings, photos, or health logs never leave their own devices.

## What problem does it solve?

Building a good AI model usually requires a large, diverse dataset. Gathering that data in one place is expensive, often legally problematic, and always a privacy risk. Hive removes that bottleneck:

- **For model builders (Host):** Start a training session with one command. Volunteers join from anywhere, train the model on their own data, and send back only a tiny mathematical summary of what they learned — not the data itself. Host authors the model, signs the `.hivepkg` package with an Ed25519 key, runs the orchestrator, and publishes a Join URI (`hive://<authority>/<session_id>?t=<join_token>&fp=<fingerprint>&tls=<0|1>`).
- **For volunteers (Contributors):** Help improve a model you care about without giving up your privacy. You decide when your computer trains (idle time only, if you like), how much CPU it uses, and for how many rounds. You can stop at any time. Thus contributor joins using the URI, inspects the signed model in their local web UI, validates private data locally inside an isolated worker sandbox, trains local epochs, and uploads only the parameter delta ($\Delta W = W_{\text{local}} - W_{\text{global}}$).

- **Security & Privacy**: Zero raw data leaves contributor machines. Packages are audited by an AST static analyzer, and model weight updates can optionally be protected with Differential Privacy (DP).

## How is it different from just uploading data?

In Hive, **your raw files never leave your device.** Only mathematical weight updates — describing what the model learned — are sent back. The host receives a collective improvement, not your personal recordings.

## What can it be used for?

- Community wildlife monitoring (bird calls, frog populations)  
- Medical research across hospitals that cannot pool patient records  
- Language models trained on private document collections  
- Any domain where data is scattered, sensitive, or legally restricted


## Quick Orientation

```
hive-fl/
├── src/hive/                       # Core Hive implementation
│   ├── cli/                        # Typer CLI commands (host, join, package, keygen, doctor)
│   ├── client/                     # Contributor runtime (agent, local UI, sandboxes, worker)
│   ├── core/                       # Data contracts, schemas, tensor math, URI parsing
│   ├── host/                       # Host server (orchestrator, public API, admin API, DB)
│   └── package/                    # .hivepkg builder, inspector, AST linting, and signing
├── examples/
│   ├── demo_pkg/                   # Sample model package (model.py, preprocess.py, manifest.json)
│   └── demo.hivepkg                # Compiled and signed demo model package
├── scripts/
│   └── run_containerized_test.sh   # Automated 1-Host + 2-Contributor containerized runner
├── data/                           # Local contributor dataset directories
├── tests/
│   ├── unit/                       # 144 unit tests covering CLI, crypto, DP, sandboxes, and APIs
│   └── integration/                # End-to-end multi-node federated round integration test
├── user-guide.md                   # Comprehensive real-world deployment & custom model guide
└── Dockerfile                      # Fast, layer-cached Docker build with PyTorch CPU wheel support
```

---

### Primary CLI Commands

| Command | Description | Example |
|---|---|---|
| `hive host` | Start a federated learning Host server | `hive host --package model.hivepkg --rounds 10 --min-clients 3` |
| `hive join` | Join an active session as a compute contributor | `hive join --uri "hive://..." --data-dir ./my_data` |
| `hive package build` | Build and cryptographically sign a `.hivepkg` | `hive package build --src ./model_dir --key key.pem --out model.hivepkg` |
| `hive keygen` | Generate an Ed25519 author signing keypair | `hive keygen -o host_key.pem` |
| `hive doctor` | Run diagnostics on Docker, Python, and ports | `hive doctor` |

---

## 2-Minute Containerized Quickstart

Test the full multi-node federated learning flow (1 Host + 2 Contributors with all three web dashboards active) in one command:

```bash
# 1. Build the fast Docker image (~2 minutes first time, ~1.2s on rebuilds)
docker build -t hive-fl:latest .

# 2. Run the automated multi-node verification
bash scripts/run_containerized_test.sh
```

To keep the containers active after training so you can explore the web dashboards in your browser:

```bash
KEEP_ALIVE=1 bash scripts/run_containerized_test.sh
```

### Active Dashboards & Port Mappings

| Role | Port | URL | Description |
|---|---|---|---|
| **Host Admin** | `8000` | `http://localhost:8000/?token=<ADMIN_TOKEN>` | Real-time session metrics, loss curves, accuracy, and participant nodes |
| **Public FL API** | `9000` | `http://localhost:9000` | Distributed fleet API endpoint for contributor connections |
| **Contributor 1** | `8765` | `http://localhost:8765/?token=<TOKEN_1>` | Local UI, AST code review, and "What Leaves My Device" transparency view |
| **Contributor 2** | `8766` | `http://localhost:8766/?token=<TOKEN_2>` | Local UI, AST code review, and hardware resource governor |

---

## Running the Test Suite

```bash
# Run all unit and integration tests (145 tests)
uv run --with pytest --with pytest-asyncio pytest tests/unit tests/integration
```


*MIT License — © 2026 Soumik Sarker*