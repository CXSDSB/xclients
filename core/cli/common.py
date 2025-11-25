from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np


def spec(tree: dict[str, jax.Array] | jax.Array):
    return jax.tree.map(
        lambda x: (x.shape, x.dtype) if isinstance(x, jax.Array | np.ndarray) else type(x), tree
    )


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8080
    show: bool = False
