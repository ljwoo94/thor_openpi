import dataclasses
import json
import logging
import pathlib
import statistics
import time
from typing import Any

import numpy as np
from openpi.policies import libero_policy
from openpi.policies import policy_config
import torch
import tyro

from openpi.training import config as _config


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    """Benchmark local pi05_libero policy inference latency."""

    # Training config name.
    config_name: str = "pi05_libero"
    # Checkpoint directory. Must contain model.safetensors for PyTorch benchmarking.
    checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"
    # PyTorch device, for example cuda, cuda:0, or cpu.
    device: str | None = None
    # Number of warmup inference calls before timing.
    warmup_iters: int = 5
    # Number of measured inference calls.
    iters: int = 50
    # Number of denoising steps passed to model.sample_actions.
    num_steps: int = 10
    # Random seed for deterministic synthetic LIBERO inputs and optional noise.
    seed: int = 0
    # Use fixed noise for every measured call to make correctness comparisons easier.
    fixed_noise: bool = True
    # Optional JSON output path.
    output_json: pathlib.Path | None = None


def _sync_if_needed(device: str | None) -> None:
    if device is not None and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _hardware_info(device: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None and device.startswith("cuda") and torch.cuda.is_available():
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                "cuda_version": torch.version.cuda,
                "gpu_name": props.name,
                "gpu_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_gb": props.total_memory / 1024**3,
            }
        )
    return info


def _make_observation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = libero_policy.make_libero_example()
    obs["observation/state"] = rng.random(8, dtype=np.float32)
    obs["observation/image"] = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    obs["observation/wrist_image"] = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    obs["prompt"] = "do something"
    return obs


def _make_noise(seed: int, action_horizon: int, action_dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    train_config = _config.get_config(args.config_name)
    sample_kwargs = {"num_steps": args.num_steps}
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        sample_kwargs=sample_kwargs,
        pytorch_device=args.device,
    )

    device = args.device
    if device is None and getattr(policy, "_is_pytorch_model", False):
        device = getattr(policy, "_pytorch_device", None)

    obs = _make_observation(args.seed)
    noise = None
    if args.fixed_noise:
        noise = _make_noise(args.seed + 1, train_config.model.action_horizon, train_config.model.action_dim)

    logger.info("Hardware: %s", json.dumps(_hardware_info(device), indent=2))
    logger.info("Warming up for %d iterations", args.warmup_iters)
    for _ in range(args.warmup_iters):
        _sync_if_needed(device)
        policy.infer(obs, noise=noise)
        _sync_if_needed(device)

    total_ms: list[float] = []
    policy_timings: dict[str, list[float]] = {}

    logger.info("Measuring %d iterations", args.iters)
    for _ in range(args.iters):
        _sync_if_needed(device)
        start = time.perf_counter()
        result = policy.infer(obs, noise=noise)
        _sync_if_needed(device)
        total_ms.append((time.perf_counter() - start) * 1000)
        for key, value in result.get("policy_timing", {}).items():
            policy_timings.setdefault(key, []).append(float(value))

    summary = {
        "config_name": args.config_name,
        "checkpoint_dir": args.checkpoint_dir,
        "warmup_iters": args.warmup_iters,
        "iters": args.iters,
        "num_steps": args.num_steps,
        "fixed_noise": args.fixed_noise,
        "hardware": _hardware_info(device),
        "total_policy_infer_ms": _stats(total_ms),
        "policy_timing_ms": {key: _stats(values) for key, values in sorted(policy_timings.items())},
    }

    print(json.dumps(summary, indent=2))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main(tyro.cli(Args))
