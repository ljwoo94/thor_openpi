from __future__ import annotations

from collections.abc import Sequence
import time
from typing import Any

import einops
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi_client import image_tools

_LIBERO_ACTION_DIM = 7


def _as_prompt(prompt: Any, default_prompt: str | None) -> str:
    if prompt is None:
        if default_prompt is None:
            raise ValueError("Prompt is required for FlashRT policy inference.")
        return default_prompt
    if isinstance(prompt, bytes):
        return prompt.decode("utf-8")
    if isinstance(prompt, np.ndarray):
        return str(prompt.item()) if prompt.ndim == 0 else str(prompt)
    return str(prompt)


def _parse_image(image: Any, *, resolution: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot often provides [0, 1] floats; already-normalized OpenPI
        # tensors may be [-1, 1]. FlashRT wants HWC uint8 or normalized fp16.
        if image.size and np.nanmin(image) < 0.0:
            image = (image + 1.0) / 2.0
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if image.shape[:2] != resolution:
        image = image_tools.resize_with_pad(image, *resolution)
    return np.ascontiguousarray(image)


class FlashRTPolicy(_base_policy.BasePolicy):
    """OpenPI policy adapter backed by FlashRT's pi0.5 runtime.

    The adapter intentionally preserves the robot-facing ``policy.infer(obs)``
    contract while routing the hot model path through FlashRT. It targets the
    LIBERO pi0.5 checkpoint shape: two image views, text prompt, and actions
    returned as ``(10, 7)`` robot-space chunks.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        default_prompt: str | None = None,
        num_views: int = 2,
        autotune: int = 3,
        hardware: str = "thor",
        framework: str = "torch",
        config: str = "pi05",
        use_fp8: bool = True,
        use_fp4: bool = False,
        recalibrate: bool = False,
        image_resolution: tuple[int, int] = (224, 224),
        metadata: dict[str, Any] | None = None,
    ):
        if num_views != 2:
            raise ValueError("FlashRTPolicy currently supports LIBERO num_views=2.")

        import flash_rt

        self._default_prompt = default_prompt
        self._num_views = num_views
        self._image_resolution = image_resolution
        self._metadata = {
            "runtime": "flashrt",
            "checkpoint_dir": str(checkpoint_dir),
            "config": config,
            "framework": framework,
            "hardware": hardware,
            "num_views": num_views,
            **(metadata or {}),
        }
        self._model = flash_rt.load_model(
            checkpoint=str(checkpoint_dir),
            config=config,
            framework=framework,
            hardware=hardware,
            num_views=num_views,
            autotune=autotune,
            use_fp8=use_fp8,
            use_fp4=use_fp4,
            recalibrate=recalibrate,
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[override]
        if noise is not None:
            raise NotImplementedError(
                "FlashRTPolicy does not yet support external noise injection. "
                "Use seeded/runtime-level comparisons or add a FlashRT noise hook first."
            )

        images = self._extract_images(obs)
        prompt = _as_prompt(obs.get("prompt"), self._default_prompt)
        state = obs.get("observation/state")

        start_time = time.monotonic()
        actions = self._model.predict(images=images, prompt=prompt, state=state)
        infer_ms = (time.monotonic() - start_time) * 1000
        actions = np.asarray(actions)
        if actions.ndim != 2:
            raise ValueError(f"Expected FlashRT actions with shape (horizon, dim), got {actions.shape}.")
        if actions.shape[-1] < _LIBERO_ACTION_DIM:
            raise ValueError(f"Expected at least {_LIBERO_ACTION_DIM} action dims, got {actions.shape[-1]}.")

        return {
            "actions": actions[:, :_LIBERO_ACTION_DIM],
            "policy_timing": {"infer_ms": infer_ms},
            "runtime": "flashrt",
        }

    def _extract_images(self, obs: dict) -> Sequence[np.ndarray]:
        if "images" in obs:
            raw_images = list(obs["images"])
            if len(raw_images) < self._num_views:
                raise ValueError(f"Expected at least {self._num_views} images, got {len(raw_images)}.")
            return [
                _parse_image(image, resolution=self._image_resolution)
                for image in raw_images[: self._num_views]
            ]

        try:
            base_image = obs["observation/image"]
            wrist_image = obs["observation/wrist_image"]
        except KeyError as exc:
            raise KeyError(
                "FlashRTPolicy expects LIBERO observation keys "
                "'observation/image' and 'observation/wrist_image'."
            ) from exc

        return [
            _parse_image(base_image, resolution=self._image_resolution),
            _parse_image(wrist_image, resolution=self._image_resolution),
        ]

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
