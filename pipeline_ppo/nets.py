"""A two-hidden-layer neural network in plain NumPy, with hand-written backprop and Adam.

No deep learning framework: every forward and backward step is visible, which is
the point of a code-along.
"""
from __future__ import annotations

import numpy as np


class MLP:
    def __init__(self, n_in: int, n_out: int, hidden: int = 64, seed: int = 0, out_scale: float = 0.01):
        rng = np.random.default_rng(seed)
        def init(i, o, s=1.0):
            return rng.normal(0, s * np.sqrt(1.0 / i), (i, o))
        self.p = {"W1": init(n_in, hidden), "b1": np.zeros(hidden),
                  "W2": init(hidden, hidden), "b2": np.zeros(hidden),
                  "W3": init(hidden, n_out, out_scale), "b3": np.zeros(n_out)}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, x: np.ndarray):
        p = self.p
        h1 = np.tanh(x @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        out = h2 @ p["W3"] + p["b3"]
        return out, (x, h1, h2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def backward(self, cache, d_out: np.ndarray) -> dict:
        """d_out = dLoss/dOutput (already averaged over the batch). Returns parameter gradients."""
        x, h1, h2 = cache
        p = self.p
        g = {"W3": h2.T @ d_out, "b3": d_out.sum(0)}
        d_h2 = (d_out @ p["W3"].T) * (1 - h2 ** 2)
        g["W2"], g["b2"] = h1.T @ d_h2, d_h2.sum(0)
        d_h1 = (d_h2 @ p["W2"].T) * (1 - h1 ** 2)
        g["W1"], g["b1"] = x.T @ d_h1, d_h1.sum(0)
        return g

    def adam(self, g: dict, lr: float = 3e-4, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, clip: float = 0.5):
        # Global gradient-norm clipping, as in most PPO implementations.
        norm = np.sqrt(sum((v ** 2).sum() for v in g.values()))
        if norm > clip:
            g = {k: v * clip / norm for k, v in g.items()}
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= lr * mh / (np.sqrt(vh) + eps)

    def copy(self) -> "MLP":
        c = MLP.__new__(MLP)
        c.p = {k: v.copy() for k, v in self.p.items()}
        c.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        c.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        c.t = 0
        return c


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))
