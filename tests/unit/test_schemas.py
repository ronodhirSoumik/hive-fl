"""
Unit tests for Pydantic protocol models and schemas.
"""

import pytest
from pydantic import ValidationError
from hive.core.schemas import (
    AggregationConfig,
    AggregationName,
    Capabilities,
    DataRequirements,
    DPPolicy,
    Heartbeat,
    Hyperparameters,
    JoinRequest,
    JoinResponse,
    Manifest,
    NodeState,
    ResourcePolicy,
    RoundInfo,
    SessionInfo,
    SessionState,
    UpdateMeta,
    ValidateResult,
    WorkerConfig,
    WorkerResult,
)


def test_capabilities_schema() -> None:
    cap = Capabilities(os="linux", cores=4, has_docker=True, has_gpu=False, ram_mb=8192)
    assert cap.cores == 4
    assert cap.has_docker is True

    # Cores must be >= 1
    with pytest.raises(ValidationError):
        Capabilities(os="linux", cores=0, has_docker=True, has_gpu=False)


def test_join_request_and_response() -> None:
    req = JoinRequest(
        nickname="AlphaNode",
        capabilities=Capabilities(os="windows", cores=8, has_docker=False, has_gpu=True),
    )
    assert req.nickname == "AlphaNode"

    res = JoinResponse(node_id="node_1", node_secret="secret_abc", approval="approved")
    assert res.approval == "approved"

    with pytest.raises(ValidationError):
        JoinResponse(node_id="node_1", node_secret="secret_abc", approval="invalid_status")  # type: ignore


def test_session_and_round_info() -> None:
    sess = SessionInfo(
        session_id="s123",
        name="Test Session",
        description="A test session",
        data_requirements="Audio files",
        package_digest="sha256:112233",
        host_key_fingerprint="ed25519:aabbcc",
        status=SessionState.RUNNING,
        total_rounds=10,
        current_round=1,
    )
    assert sess.status == SessionState.RUNNING

    rnd = RoundInfo(
        status="open",
        round_id=1,
        deadline_ts=1700000000.0,
        model_hash="sha256:model1",
        dp_policy=DPPolicy(enabled=True, clip_norm=2.5, noise_multiplier=0.8),
    )
    assert rnd.status == "open"
    assert rnd.dp_policy.enabled is True
    assert rnd.dp_policy.clip_norm == 2.5


def test_update_meta_schema() -> None:
    meta = UpdateMeta(
        round_id=2,
        base_model_hash="sha256:base",
        num_train_samples=150,
        train_loss=0.32,
        eval_loss=0.29,
        eval_metrics={"accuracy": 0.91},
        local_epochs=2,
        duration_s=5.2,
    )
    assert meta.num_train_samples == 150
    assert meta.eval_metrics["accuracy"] == 0.91

    with pytest.raises(ValidationError):
        # num_train_samples must be >= 1
        UpdateMeta(
            round_id=2,
            base_model_hash="sha256:base",
            num_train_samples=0,
            train_loss=0.32,
            eval_loss=0.29,
        )


def test_worker_config_and_result() -> None:
    wcfg = WorkerConfig(task="train", max_cpu_percent=60)
    assert wcfg.task == "train"
    assert wcfg.max_cpu_percent == 60

    wres = WorkerResult(status="ok", num_train_samples=100, train_loss=0.2)
    assert wres.status == "ok"
    assert wres.num_train_samples == 100

    val_res = ValidateResult(ok=True, num_samples=300)
    assert val_res.ok is True
    assert val_res.num_samples == 300


def test_manifest_schema() -> None:
    manifest = Manifest(
        package_name="bird-classifier",
        version="0.1.0",
        description="Classifies chirps",
        model_entrypoint="model.py:BirdClassifier",
        preprocessor_entrypoint="preprocess.py:AudioPreprocessor",
        data_requirements=DataRequirements(description="WAV audio", formats=[".wav"], min_samples=10),
    )
    assert manifest.package_name == "bird-classifier"
    assert manifest.loss == "cross_entropy"
    assert manifest.metrics == "accuracy"
    assert manifest.data_requirements.min_samples == 10