"""
In-Sandbox Worker training and validation engine.
Dynamically loads Host-supplied models and preprocessors from the job directory,
executes local epochs with CPU throttling, applies DP noise, and outputs updates.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hive.core.schemas import Manifest, WorkerConfig, WorkerResult
from hive.core.tensors import (
    add_gaussian_noise,
    bytes_to_state_dict,
    clip_by_norm,
    compute_delta,
    state_dict_to_bytes,
)


def load_module_from_file(file_path: Path | str, module_name: str) -> Any:
    """Dynamically imports a Python file without requiring it to be in sys.path."""
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Source file not found: {p}")
    spec = importlib.util.spec_from_file_location(module_name, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module specification for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_entrypoint(pkg_dir: Path, entrypoint: str) -> Any:
    """Resolves an entrypoint like 'model.py:BirdClassifier' into the callable/class."""
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint format: '{entrypoint}'. Expected 'filename.py:ClassName'")
    filename, attr_name = entrypoint.split(":", 1)
    file_path = pkg_dir / filename
    mod = load_module_from_file(file_path, f"hive_pkg_{Path(filename).stem}")
    if not hasattr(mod, attr_name):
        raise AttributeError(f"Module '{filename}' has no attribute '{attr_name}'")
    return getattr(mod, attr_name)


def get_loss_fn(loss_name: str) -> nn.Module:
    """Returns loss criterion matching manifest loss name."""
    clean = loss_name.lower().strip()
    if clean == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif clean == "mse":
        return nn.MSELoss()
    elif clean == "bce_logits":
        return nn.BCEWithLogitsLoss()
    raise ValueError(f"Unsupported loss function: '{loss_name}'")


def run_worker_task(job_dir: Path | str) -> None:
    """
    Main in-worker execution orchestrator.
    Reads <job>/in/config.json and runs validate or train task.
    """
    job = Path(job_dir).resolve()
    in_dir = job / "in"
    out_dir = job / "out"
    pkg_dir = in_dir / "pkg"
    progress_file = out_dir / "progress.jsonl"
    result_file = out_dir / "result.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    try:
        # 1. Read config and manifest
        cfg_file = in_dir / "config.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Missing {cfg_file}")
        cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8"))
        worker_config = WorkerConfig(**cfg_dict)
        data_dir = cfg_dict.get("data_dir", "")

        # Set deterministic seeds
        seed = worker_config.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        manifest_file = pkg_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Missing {manifest_file}")
        manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))

        # 2. Resolve Preprocessor
        preprocessor_cls = resolve_entrypoint(pkg_dir, manifest.preprocessor_entrypoint)
        preprocessor = preprocessor_cls()

        # ---------------- Task: VALIDATE ----------------
        if worker_config.task == "validate":
            val_res = preprocessor.validate_local_directory(data_dir)
            # Write detailed message only to a local-only file (never sent over the wire)
            local_msg_file = out_dir / "local_message.txt"
            local_msg_file.write_text(val_res.message or ("OK" if val_res.ok else "Validation failed"), encoding="utf-8")

            if not val_res.ok:
                res = WorkerResult(
                    status="error",
                    error_code="ERR_DATA_VALIDATION_FAILED",
                    error_message=None,  # Never leak local path/details into result.json
                )
            else:
                res = WorkerResult(
                    status="ok",
                    num_train_samples=val_res.num_samples,
                    num_eval_samples=0,
                    duration_s=time.time() - start_time,
                )
            result_file.write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")
            return

        # ---------------- Task: TRAIN ----------------
        elif worker_config.task == "train":
            # Apply CPU thread throttling
            cores = psutil.cpu_count(logical=True) or 2
            threads = max(1, math.ceil(cores * (worker_config.max_cpu_percent / 100.0)))
            torch.set_num_threads(threads)

            # Load global weights
            global_file = in_dir / "global.safetensors"
            if not global_file.exists():
                raise FileNotFoundError("Missing global.safetensors in job directory")
            global_weights = bytes_to_state_dict(global_file.read_bytes())

            # Instantiate model
            model_cls = resolve_entrypoint(pkg_dir, manifest.model_entrypoint)
            model: nn.Module = model_cls()
            model.load_state_dict(global_weights, strict=False)

            # Device selection: cpu | cuda | mps
            device_type = "cpu"
            if worker_config.use_gpu:
                if torch.cuda.is_available():
                    device_type = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device_type = "mps"
            device = torch.device(device_type)
            model.to(device)

            # Build datasets
            train_ds = preprocessor.build_dataset(data_dir, split="train", seed=worker_config.seed)
            eval_ds = preprocessor.build_dataset(data_dir, split="eval", seed=worker_config.seed)

            num_train = len(train_ds) if hasattr(train_ds, "__len__") else 0
            num_eval = len(eval_ds) if hasattr(eval_ds, "__len__") else 0

            batch_size = manifest.hyperparameters.batch_size
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False) if num_eval > 0 else None

            # Loss and optimizer
            criterion = get_loss_fn(manifest.loss)
            lr = manifest.hyperparameters.learning_rate
            if manifest.hyperparameters.optimizer == "sgd":
                optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Local training epochs
            model.train()
            train_loss_accum = 0.0
            total_train_batches = 0

            with open(progress_file, "a", encoding="utf-8") as prog_f:
                for epoch in range(1, manifest.hyperparameters.local_epochs + 1):
                    for b_idx, (inputs, targets) in enumerate(train_loader):
                        t_b_start = time.perf_counter()

                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                        t_b_elapsed = time.perf_counter() - t_b_start

                        # CPU Duty-Cycle Throttle Sleep
                        if worker_config.max_cpu_percent < 100:
                            # Sleep so that t_active / (t_active + t_sleep) <= max_cpu_pct
                            duty_ratio = worker_config.max_cpu_percent / 100.0
                            sleep_s = max(0.0, t_b_elapsed * ((1.0 / duty_ratio) - 1.0))
                            if sleep_s > 0:
                                time.sleep(sleep_s)

                        loss_val = float(loss.item())
                        train_loss_accum += loss_val
                        total_train_batches += 1

                        # Emit progress record
                        prog_record = {
                            "ts": time.time(),
                            "phase": "train",
                            "epoch": epoch,
                            "batch": b_idx,
                            "loss": round(loss_val, 4),
                        }
                        prog_f.write(json.dumps(prog_record) + "\n")
                        prog_f.flush()

            avg_train_loss = (train_loss_accum / max(1, total_train_batches))

            # Evaluation pass
            model.eval()
            eval_loss_accum = 0.0
            total_eval_batches = 0
            correct_eval = 0
            total_eval_samples = 0

            if eval_loader:
                with torch.no_grad():
                    for inputs, targets in eval_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        e_loss = criterion(outputs, targets)
                        eval_loss_accum += float(e_loss.item())
                        total_eval_batches += 1

                        if manifest.metrics == "accuracy":
                            preds = outputs.argmax(dim=-1)
                            correct_eval += int((preds == targets).sum().item())
                            total_eval_samples += targets.size(0)

            avg_eval_loss = (eval_loss_accum / max(1, total_eval_batches)) if total_eval_batches > 0 else avg_train_loss
            eval_acc = (correct_eval / max(1, total_eval_samples)) if total_eval_samples > 0 else 0.0

            # Compute weight delta: local - global
            local_weights = {k: v.detach().cpu().to(torch.float32) for k, v in model.state_dict().items()}
            delta = compute_delta(local_weights, global_weights)

            # Apply Differential Privacy if enabled
            if worker_config.dp and worker_config.dp.enabled:
                delta, _ = clip_by_norm(delta, worker_config.dp.clip_norm)
                delta = add_gaussian_noise(delta, worker_config.dp.clip_norm, worker_config.dp.noise_multiplier)

            # Save update.safetensors
            update_file = out_dir / "update.safetensors"
            update_file.write_bytes(state_dict_to_bytes(delta))

            # Save result.json
            res = WorkerResult(
                status="ok",
                num_train_samples=num_train,
                num_eval_samples=num_eval,
                train_loss=round(avg_train_loss, 4),
                eval_loss=round(avg_eval_loss, 4),
                eval_metrics={"accuracy": round(eval_acc, 4)} if manifest.metrics == "accuracy" else {},
                duration_s=round(time.time() - start_time, 2),
            )
            result_file.write_text(json.dumps(res.model_dump(), indent=2), encoding="utf-8")

        else:
            raise ValueError(f"Unknown worker task: '{worker_config.task}'")

    except Exception as e:
        import traceback
        log_file = out_dir / "worker.log"
        log_file.write_text(traceback.format_exc(), encoding="utf-8")
        sys.stderr.write(f"Worker failure: {e}\n")
        err_res = WorkerResult(
            status="error",
            error_code="ERR_WORKER_EXECUTION",
            error_message=None,
            duration_s=round(time.time() - start_time, 2),
        )
        result_file.write_text(json.dumps(err_res.model_dump(), indent=2), encoding="utf-8")