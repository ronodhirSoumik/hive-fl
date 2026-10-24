# Hive

**Train smarter AI models together — without sharing your data.**

---

Hive is an open-source tool that lets people collaborate to train machine learning models, even when their data is private and cannot be shared. Think of it like a community science project: many volunteers contribute computing time and local expertise, but their raw recordings, photos, or health logs never leave their own devices.

## What problem does it solve?

Building a good AI model usually requires a large, diverse dataset. Gathering that data in one place is expensive, often legally problematic, and always a privacy risk. Hive removes that bottleneck:

- **For model builders (Hosts):** Start a training session with one command. Volunteers join from anywhere, train the model on their own data, and send back only a tiny mathematical summary of what they learned — not the data itself.  
- **For volunteers (Contributors):** Help improve a model you care about without giving up your privacy. You decide when your computer trains (idle time only, if you like), how much CPU it uses, and for how many rounds. You can stop at any time.

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
├── README.md               ← You are here (non-technical overview)
├── documentation.md        ← Developer & user documentation (updated every phase)
├── architecture.md         ← Living technical architecture document
├── pyproject.toml          ← Project configuration, dependencies, and CLI entrypoint
├── Makefile                ← Standard developer targets (install, test, lint, demo)
├── docs/                   ← Protocol specifications and ADR decisions
│   ├── PROTOCOL.md         ← Wire protocol specification for Public FL API (/v1)
│   └── DECISIONS.md        ← Architecture Decision Records index
├── src/hive/               ← Core platform source code
│   ├── core/               ← Config, schemas, contracts, errors, URI, logging, hashing, tensors, dp
│   ├── package/            ← .hivepkg builder, reader, signing, linting
│   ├── host/               ← Host server, orchestrator, state DB, aggregators, admin API, tunnel
│   ├── client/             ← Contributor agent, resource governor, sandbox executor, worker
│   ├── cli/                ← Hive unified CLI commands
│   └── demo/               ← Built-in BirdSound-Lite demo package & data generator
└── tests/                  ← Unit, integration, and end-to-end tests
    ├── unit/
    ├── integration/
    └── e2e/
```

&nbsp;

---

*MIT License — © 2026 Soumik Sarker*