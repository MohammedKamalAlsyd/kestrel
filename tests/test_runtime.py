"""Tests for `kestrel.utils.train`, `kestrel.utils.data`, and `kestrel.utils.env`."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from kestrel import utils
from kestrel.utils.data import normalize_words
from kestrel.utils.env import find_repo_root
from kestrel.utils.train import (
    RuntimeEnv,
    get_device_info,
    get_dtype,
    is_bf16_supported,
    is_dtype_supported,
    is_fp16_supported,
    make_autocast_context,
    make_grad_scaler,
    resolve_device,
    resolve_dtype,
    seed_everything,
    setup_runtime,
)


def test_namespace_imports_and_repo_root() -> None:
    assert utils.train.resolve_device is resolve_device
    assert utils.data.normalize_words is normalize_words
    assert utils.env.find_repo_root is find_repo_root
    assert normalize_words("Hello, Kestrel 350M!") == ["hello", "kestrel", "m"]
    assert find_repo_root() == Path(__file__).resolve().parents[1]


def test_resolve_device_explicit_cpu() -> None:
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_matches_hardware(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("kestrel.utils.train.device.is_mps_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")

    monkeypatch.setattr("kestrel.utils.train.device.is_mps_available", lambda: True)
    assert resolve_device("auto") == torch.device("mps")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto", local_rank=1) == torch.device("cuda:1")


def test_resolve_device_cuda_fallback_and_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("kestrel.utils.train.device.is_mps_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="CUDA is unavailable"):
        assert resolve_device("cuda", fallback=True) == torch.device("cpu")

    with pytest.raises(RuntimeError, match="CUDA device requested"):
        resolve_device("cuda", fallback=False)


def test_resolve_dtype_cuda_ampere_vs_t4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=False: True)
    assert is_bf16_supported("cuda:0") is True
    assert resolve_dtype("auto", "cuda:0") == ("bf16", torch.bfloat16)
    assert get_dtype("bf16", "cuda:0") == torch.bfloat16

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=False: False)
    assert is_bf16_supported("cuda:0") is False
    assert is_fp16_supported("cuda:0") is True
    assert resolve_dtype("auto", "cuda:0") == ("fp16", torch.float16)

    with pytest.warns(RuntimeWarning, match="falling back to 'fp16'"):
        assert resolve_dtype("bf16", "cuda:0", fallback=True) == ("fp16", torch.float16)

    with pytest.raises(RuntimeError, match="not natively supported"):
        resolve_dtype("bf16", "cuda:0", fallback=False)


def test_resolve_dtype_cpu_and_aliases() -> None:
    assert resolve_dtype("auto", "cpu") == ("fp32", torch.float32)
    assert resolve_dtype("float32", "cpu") == ("fp32", torch.float32)
    assert resolve_dtype(torch.float32, "cpu") == ("fp32", torch.float32)
    assert is_fp16_supported("cpu") is False
    assert is_dtype_supported("fp32", "cpu") is True

    with pytest.warns(RuntimeWarning, match="falling back to 'fp32'"):
        assert resolve_dtype("fp16", "cpu", fallback=True) == ("fp32", torch.float32)

    with pytest.raises(RuntimeError, match="not supported"):
        resolve_dtype("fp16", "cpu", fallback=False)

    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_dtype("int8", "cpu")


def test_autocast_and_grad_scaler() -> None:
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")

    assert make_grad_scaler(cpu, torch.float32) is None
    assert make_grad_scaler(cuda, torch.bfloat16) is None
    scaler = make_grad_scaler(cuda, torch.float16)
    assert isinstance(scaler, torch.amp.GradScaler)
    assert scaler.is_enabled()

    ctx_factory = make_autocast_context(cpu, torch.float32)
    with ctx_factory():
        x = torch.ones(2, 2)
        assert x.dtype == torch.float32


def test_seed_everything_and_setup_runtime_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    assert seed_everything(100, rank=3) == 103
    a = torch.rand(4)
    seed_everything(100, rank=3)
    b = torch.rand(4)
    assert torch.equal(a, b)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        env = setup_runtime(seed=7, precision="auto", device="cpu", distributed=False)

    assert isinstance(env, RuntimeEnv)
    assert env.device == torch.device("cpu")
    assert env.dtype == torch.float32
    assert env.precision == "fp32"
    assert env.autocast_enabled is False
    assert env.scaler is None
    assert env.rank == 0
    assert env.world_size == 1
    assert env.is_main is True
    with env.autocast():
        pass
    assert "Device=cpu" in env.summary()


def test_get_device_info_returns_expected_keys() -> None:
    info = get_device_info("cpu")
    assert info["device"] == "cpu"
    assert info["device_type"] == "cpu"
    assert info["recommended_precision"] == "fp32"
    assert info["recommended_dtype"] == "torch.float32"
