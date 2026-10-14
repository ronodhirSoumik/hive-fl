"""
Pydantic data models and enums for Hive cross-boundary protocol contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"


class NodeState(str, Enum):
    REGISTERED = "registered"
    APPROVED = "approved"
    ELIGIBLE = "eligible"
    TRAINING = "training"
    PAUSED_LOCAL = "paused_local"
    OFFLINE = "offline"
    LEFT = "left"


class AggregationName(str, Enum):
    FEDAVG = "fedavg"
    TRIMMED_MEAN = "trimmed-mean"
    MEDIAN = "median"
    KRUM = "krum"
    MULTI_KRUM = "multi-krum"


class SandboxMode(str, Enum):
    AUTO = "auto"
    DOCKER = "docker"
    PROCESS = "process"


class RoundStatus(str, Enum):
    NONE = "none"
    OPEN = "open"
    CLOSED = "closed"


# ---------------------------------------------------------------------------
# Protocol Models
# ---------------------------------------------------------------------------

class Capabilities(BaseModel):
    """Contributor node hardware capabilities and platform info."""
    os: str = Field(..., description="Operating system identifier (e.g. windows, linux, darwin)")
    cores: int = Field(..., ge=1, description="Physical/logical CPU cores count")
    has_docker: bool = Field(..., description="Whether Docker daemon is running and accessible")
    has_gpu: bool = Field(..., description="Whether CUDA/MPS GPU acceleration is available")
    ram_mb: int | None = Field(default=None, ge=1, description="Total system physical memory in MB")


class JoinRequest(BaseModel):
    """Request payload for POST /v1/join."""
    nickname: str = Field(..., min_length=1, max_length=64, description="Contributor nickname")
    capabilities: Capabilities


class JoinResponse(BaseModel):
    """Response returned upon POST /v1/join."""
    node_id: str
    node_secret: str
    approval: Literal["approved", "pending"]


class SessionInfo(BaseModel):
    """Public session information returned by GET /v1/session."""
    session_id: str
    name: str
    description: str
    data_requirements: str
    package_digest: str
    host_key_fingerprint: str
    status: SessionState | str
    total_rounds: int
    current_round: int


class DPPolicy(BaseModel):
    """Differential privacy parameters."""
    enabled: bool = False
    clip_norm: float = Field(default=1.0, gt=0.0)
    noise_multiplier: float = Field(default=0.0, ge=0.0)
    delta: float = Field(default=1e-5, gt=0.0, lt=1.0)


class RoundInfo(BaseModel):
    """Current round status returned by GET /v1/round/current."""
    status: Literal["none", "open", "closed"]
    round_id: int
    deadline_ts: float | None = None
    model_hash: str | None = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    dp_policy: DPPolicy | None = None
    retry_after_s: int | None = None


class UpdateMeta(BaseModel):
    """Metadata JSON payload for POST /v1/round/{id}/update multipart upload."""
    round_id: int
    base_model_hash: str
    num_train_samples: int = Field(..., ge=1)
    num_eval_samples: int = Field(default=0, ge=0)
    train_loss: float
    eval_loss: float
    eval_metrics: dict[str, float] = Field(default_factory=dict)
    local_epochs: int = Field(default=1, ge=1)
    duration_s: float = Field(default=0.0, ge=0.0)
    delta_sha256: str = ""
    dp: dict[str, Any] = Field(default_factory=dict)


class Heartbeat(BaseModel):
    """Heartbeat payload sent by client to POST /v1/heartbeat."""
    state: Literal["waiting", "training", "uploading", "paused_local"]
    reason: str | None = None
    progress: float | None = Field(default=None, ge=0.0, le=1.0)


class WorkerConfig(BaseModel):
    """Job configuration written to job/in/config.json for the sandbox worker."""
    task: Literal["validate", "train"]
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    dp: DPPolicy | None = None
    max_cpu_percent: int = Field(default=100, ge=1, le=100)
    seed: int = 42
    use_gpu: bool = False


class WorkerResult(BaseModel):
    """Execution output written to job/out/result.json by the worker."""
    status: Literal["ok", "error"]
    num_train_samples: int = 0
    num_eval_samples: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    eval_metrics: dict[str, float] = Field(default_factory=dict)
    duration_s: float = 0.0
    error_code: str | None = None
    error_message: str | None = None


class ValidateResult(BaseModel):
    """Output for data validation task."""
    ok: bool
    num_samples: int = 0
    message: str = ""


class ResourcePolicy(BaseModel):
    """Client resource governor policy."""
    only_when_idle: bool = True
    idle_cpu_threshold: float = Field(default=25.0, ge=1.0, le=100.0)
    idle_minutes: float = Field(default=2.0, ge=0.0)
    max_cpu_percent: int = Field(default=50, ge=1, le=100)
    max_ram_mb: int = Field(default=2048, ge=256)
    use_gpu: bool = False
    pause_on_battery: bool = True
    schedule: dict[str, Any] = Field(default_factory=dict)
    max_rounds: int = 0
    auto_rejoin: bool = True


class AggregationConfig(BaseModel):
    """Server-side aggregation configuration."""
    name: AggregationName = AggregationName.FEDAVG
    server_clip_norm: float | None = None
    trim_ratio: float = Field(default=0.2, ge=0.0, lt=0.5)
    max_byzantine: int | None = None


class Hyperparameters(BaseModel):
    """Training hyperparameters specified in manifest.json."""
    local_epochs: int = Field(default=2, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimizer: Literal["sgd", "adam"] = "adam"
    eval_split: float = Field(default=0.2, ge=0.0, lt=1.0)


class DataRequirements(BaseModel):
    """Data specification for contributor datasets."""
    description: str
    formats: list[str] = Field(default_factory=list)
    min_samples: int = Field(default=1, ge=1)


class Manifest(BaseModel):
    """Manifest describing a .hivepkg package contents and training contract."""
    package_name: str
    version: str
    description: str
    framework: str = "pytorch"
    min_framework_version: str = "2.0.0"
    model_entrypoint: str
    preprocessor_entrypoint: str
    loss: str = "cross_entropy"
    metrics: str = "accuracy"
    hyperparameters: Hyperparameters = Field(default_factory=Hyperparameters)
    data_requirements: DataRequirements
    dp_suggested: dict[str, Any] = Field(default_factory=dict)
    content_digest: str = ""