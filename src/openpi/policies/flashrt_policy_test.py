import sys
import types

import numpy as np
import pytest

from openpi.policies import flashrt_policy


class _FakeFlashRTModel:
    def __init__(self):
        self.predict_calls = []

    def predict(self, *, images, prompt, state=None):
        self.predict_calls.append({"images": images, "prompt": prompt, "state": state})
        return np.arange(320, dtype=np.float32).reshape(10, 32)


def _install_fake_flash_rt(monkeypatch):
    fake_model = _FakeFlashRTModel()
    load_calls = []

    def load_model(**kwargs):
        load_calls.append(kwargs)
        return fake_model

    fake_module = types.SimpleNamespace(load_model=load_model)
    monkeypatch.setitem(sys.modules, "flash_rt", fake_module)
    return fake_model, load_calls


def test_flashrt_policy_loads_pi05_thor_backend(monkeypatch):
    _, load_calls = _install_fake_flash_rt(monkeypatch)

    flashrt_policy.FlashRTPolicy("/tmp/pi05_libero", autotune=5, use_fp4=True)

    assert load_calls == [
        {
            "checkpoint": "/tmp/pi05_libero",
            "config": "pi05",
            "framework": "torch",
            "hardware": "thor",
            "num_views": 2,
            "autotune": 5,
            "use_fp8": True,
            "use_fp4": True,
            "recalibrate": False,
        }
    ]


def test_flashrt_policy_infer_accepts_libero_observation(monkeypatch):
    fake_model, _ = _install_fake_flash_rt(monkeypatch)
    policy = flashrt_policy.FlashRTPolicy("/tmp/pi05_libero")
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "pick up the block",
    }

    result = policy.infer(obs)

    assert result["actions"].shape == (10, 7)
    assert result["runtime"] == "flashrt"
    assert result["policy_timing"]["infer_ms"] >= 0
    assert len(fake_model.predict_calls) == 1
    call = fake_model.predict_calls[0]
    assert call["prompt"] == "pick up the block"
    np.testing.assert_array_equal(call["state"], obs["observation/state"])
    assert len(call["images"]) == 2
    np.testing.assert_array_equal(call["images"][0], obs["observation/image"])
    np.testing.assert_array_equal(call["images"][1], obs["observation/wrist_image"])
    np.testing.assert_array_equal(result["actions"], np.arange(320, dtype=np.float32).reshape(10, 32)[:, :7])


def test_flashrt_policy_infer_uses_default_prompt(monkeypatch):
    fake_model, _ = _install_fake_flash_rt(monkeypatch)
    policy = flashrt_policy.FlashRTPolicy("/tmp/pi05_libero", default_prompt="default task")
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
    }

    policy.infer(obs)

    assert fake_model.predict_calls[0]["prompt"] == "default task"


def test_flashrt_policy_rejects_noise_until_flashrt_exposes_noise_hook(monkeypatch):
    _install_fake_flash_rt(monkeypatch)
    policy = flashrt_policy.FlashRTPolicy("/tmp/pi05_libero")
    obs = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the block",
    }

    with pytest.raises(NotImplementedError, match="noise"):
        policy.infer(obs, noise=np.zeros((10, 32), dtype=np.float32))
