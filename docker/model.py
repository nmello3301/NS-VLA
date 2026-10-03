"""JAX-based NS-VLA inference model.

The upstream NS-VLA weights are not yet released (the repository's model
modules are stubbed with "Code will be released upon paper acceptance").
This module provides a drop-in JAX wrapper with the correct I/O contract
so the surrounding Docker API, benchmarks, and clients can be developed
and tested now. Swap `NSVLAJaxModel._forward` for the real network once
weights are published.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

# Fixed primitive vocabulary matching nsvla/primitives/primitive_set.py intent.
PRIMITIVES: List[str] = [
    "pick",
    "place_on",
    "place_in",
    "push",
    "pull",
    "open",
    "close",
    "rotate",
    "move_to",
    "release",
]

# A VLA action vector: [dx, dy, dz, droll, dpitch, dyaw, gripper].
ACTION_DIM = 7
IMAGE_SIZE = 224


@dataclass
class PredictionResult:
    actions: np.ndarray            # (horizon, ACTION_DIM)
    primitive: str
    primitive_logits: np.ndarray   # (len(PRIMITIVES),)
    latency_ms: float


class NSVLAJaxModel:
    """Deterministic JAX placeholder for the NS-VLA policy.

    Behavior is deterministic given (image, instruction) so clients can
    write reproducible tests against the API before real weights land.
    """

    def __init__(self, horizon: int = 8, seed: int = 0):
        self.horizon = horizon
        self._rng = jax.random.PRNGKey(seed)
        self._forward_jit = jax.jit(self._forward, static_argnames=("horizon",))

    @staticmethod
    def _preprocess_image(image: Image.Image) -> jnp.ndarray:
        img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
        arr = jnp.asarray(np.asarray(img, dtype=np.float32) / 255.0)
        return arr  # (H, W, 3)

    @staticmethod
    def _hash_instruction(instruction: str) -> jnp.ndarray:
        # Stable 32-dim feature from the instruction string.
        h = np.zeros(32, dtype=np.float32)
        for i, ch in enumerate(instruction.encode("utf-8")):
            h[i % 32] += (ch % 97) / 97.0
        norm = np.linalg.norm(h) or 1.0
        return jnp.asarray(h / norm)

    @staticmethod
    def _forward(image: jnp.ndarray, text_feat: jnp.ndarray, horizon: int):
        # Visual summary: mean pooling into a 32-dim projection.
        img_feat = jnp.mean(image.reshape(-1, 3), axis=0)        # (3,)
        img_proj = jnp.tile(img_feat, 11)[:32]                   # (32,)
        joint = 0.5 * img_proj + 0.5 * text_feat                 # (32,)

        # Symbolic head: linear projection -> primitive logits.
        W_prim = jnp.linspace(-1.0, 1.0, 32 * len(PRIMITIVES)).reshape(
            32, len(PRIMITIVES)
        )
        primitive_logits = joint @ W_prim                        # (P,)

        # Action head: linear projection -> (horizon, ACTION_DIM).
        W_act = jnp.linspace(-0.1, 0.1, 32 * horizon * ACTION_DIM).reshape(
            32, horizon * ACTION_DIM
        )
        actions = (joint @ W_act).reshape(horizon, ACTION_DIM)
        # Keep gripper channel in [0, 1].
        actions = actions.at[:, 6].set(jax.nn.sigmoid(actions[:, 6]))
        return actions, primitive_logits

    def predict(self, image: Image.Image, instruction: str) -> PredictionResult:
        start = time.perf_counter()
        img_jnp = self._preprocess_image(image)
        txt_jnp = self._hash_instruction(instruction)
        actions, logits = self._forward_jit(img_jnp, txt_jnp, horizon=self.horizon)
        actions_np = np.asarray(actions)
        logits_np = np.asarray(logits)
        primitive = PRIMITIVES[int(np.argmax(logits_np))]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return PredictionResult(
            actions=actions_np,
            primitive=primitive,
            primitive_logits=logits_np,
            latency_ms=latency_ms,
        )


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))
