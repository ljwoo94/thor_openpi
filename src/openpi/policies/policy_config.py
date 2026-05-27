import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.policies.flashrt_policy as _flashrt_policy
import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def create_flashrt_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    default_prompt: str | None = None,
    num_views: int = 2,
    autotune: int = 3,
    hardware: str = "thor",
    framework: str = "torch",
    use_fp8: bool = True,
    use_fp4: bool = False,
    recalibrate: bool = False,
) -> _flashrt_policy.FlashRTPolicy:
    """Create a FlashRT-backed policy with the same robot-facing infer API.

    This is intentionally separate from create_trained_policy so the baseline
    OpenPI/JAX/PyTorch path remains unchanged while FlashRT is validated.
    """
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))
    if train_config.name != "pi05_libero":
        raise ValueError(f"FlashRT policy currently supports only pi05_libero, got {train_config.name!r}.")
    if train_config.model.action_horizon != 10:
        raise ValueError("FlashRT pi05 LIBERO policy expects action_horizon=10.")
    if train_config.model.action_dim != 32:
        raise ValueError("FlashRT pi05 LIBERO policy expects action_dim=32.")
    if getattr(train_config.model, "discrete_state_input", None):
        raise ValueError("FlashRT pi05 LIBERO adapter expects discrete_state_input=False.")

    return _flashrt_policy.FlashRTPolicy(
        checkpoint_dir,
        default_prompt=default_prompt,
        num_views=num_views,
        autotune=autotune,
        hardware=hardware,
        framework=framework,
        config="pi05",
        use_fp8=use_fp8,
        use_fp4=use_fp4,
        recalibrate=recalibrate,
        metadata=train_config.policy_metadata,
    )
