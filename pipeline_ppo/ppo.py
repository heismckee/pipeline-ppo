"""PPO for the pipeline, written to mirror RLHF step for step.

RLHF                                 | This project
-------------------------------------|-----------------------------------------------
Supervised fine-tuned (SFT) model    | Policy cloned from the current sales playbook
Reward model score                   | Closed revenue minus play costs
Value head V(s)                      | Opportunity value: expected net revenue from here
KL penalty to the SFT model          | KL penalty to the playbook clone
PPO clipped policy update            | Same
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import N_ACTIONS, OBS_DIM, PipelineEnv, playbook_action
from .nets import MLP, log_softmax, softmax


@dataclass
class PPOConfig:
    n_deals: int = 256          # deals simulated in parallel
    horizon: int = 32           # weeks per rollout
    iterations: int = 120
    epochs: int = 4
    minibatch: int = 2048
    gamma: float = 0.99         # weekly discount
    lam: float = 0.95           # GAE lambda
    clip: float = 0.2           # PPO clip range (epsilon)
    vf_coef: float = 1.0
    ent_coef: float = 0.01
    kl_coef: float = 0.02       # beta: pull toward the playbook clone (the "reference model")
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    hidden: int = 64
    seed: int = 0


@dataclass
class History:
    rows: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1: "supervised fine-tuning" - clone the current playbook
# ---------------------------------------------------------------------------
def behavior_clone(actor: MLP, steps: int = 300, n_deals: int = 512, seed: int = 1, lr: float = 1e-3, smoothing: float = 0.2):
    """Train the actor to imitate the playbook with cross-entropy, like SFT before RLHF.

    Label smoothing keeps a little probability on other plays, so PPO still has room to explore.
    """
    env = PipelineEnv(n_deals, seed=seed)
    obs = env.reset()
    for _ in range(steps):
        a = playbook_action(obs)
        logits, cache = actor.forward(obs)
        target = np.full_like(logits, smoothing / N_ACTIONS)
        target[np.arange(len(a)), a] += 1.0 - smoothing
        d = softmax(logits) - target                # dCE/dlogits = pi - target
        actor.adam(actor.backward(cache, d / len(a)), lr=lr, clip=1.0)
        obs, _, _, _ = env.step(a)
    acc = (actor(obs).argmax(1) == playbook_action(obs)).mean()
    return float(acc)


# ---------------------------------------------------------------------------
# Step 2: collect experience
# ---------------------------------------------------------------------------
def rollout(env: PipelineEnv, obs: np.ndarray, actor: MLP, critic: MLP, horizon: int, rng: np.random.Generator):
    n = env.n
    buf = {k: [] for k in ["obs", "act", "logp", "val", "rew", "done"]}
    done_rev, done_cost = [], []
    for _ in range(horizon):
        logits = actor(obs)
        logp_all = log_softmax(logits)
        p = np.exp(logp_all)
        a = (p.cumsum(1) > rng.random((n, 1))).argmax(1)      # sample one action per deal
        v = critic(obs)[:, 0]
        nxt, r, d, info = env.step(a)
        buf["obs"].append(obs); buf["act"].append(a); buf["logp"].append(logp_all[np.arange(n), a])
        buf["val"].append(v); buf["rew"].append(r); buf["done"].append(d.astype(float))
        obs = nxt
    last_val = critic(obs)[:, 0]
    return {k: np.array(v) for k, v in buf.items()}, last_val, obs


# ---------------------------------------------------------------------------
# Step 3: advantages with GAE
# ---------------------------------------------------------------------------
def gae(rew, val, done, last_val, gamma: float, lam: float):
    """Generalized advantage estimation.

    delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    A_t     = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
    The value target ("return") is A_t + V(s_t).
    """
    T = len(rew)
    adv = np.zeros_like(rew)
    nxt_adv = np.zeros_like(last_val)
    nxt_val = last_val
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_val * nonterm - val[t]
        nxt_adv = delta + gamma * lam * nonterm * nxt_adv
        adv[t] = nxt_adv
        nxt_val = val[t]
    return adv, adv + val


# ---------------------------------------------------------------------------
# Step 4: the PPO update
# ---------------------------------------------------------------------------
def ppo_update(actor: MLP, critic: MLP, ref: MLP, batch: dict, cfg: PPOConfig, rng: np.random.Generator):
    N = len(batch["act"])
    stats = {"clip_frac": [], "approx_kl": [], "entropy": [], "kl_ref": [], "value_loss": []}
    for _ in range(cfg.epochs):
        idx = rng.permutation(N)
        for s in range(0, N, cfg.minibatch):
            j = idx[s:s + cfg.minibatch]
            o, a, old_logp = batch["obs"][j], batch["act"][j], batch["logp"][j]
            A = batch["adv"][j]
            A = (A - A.mean()) / (A.std() + 1e-8)
            R = batch["ret"][j]
            B = len(j)

            # ---- policy (actor) ----
            logits, cache = actor.forward(o)
            logp_all = log_softmax(logits)
            pi = np.exp(logp_all)
            logp = logp_all[np.arange(B), a]
            ratio = np.exp(logp - old_logp)
            clipped = np.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)
            # Objective = min(ratio*A, clip(ratio)*A). Gradient flows only where the unclipped term is the min.
            use_unclipped = (ratio * A) <= (clipped * A)
            g_logp = np.where(use_unclipped, ratio * A, 0.0)     # d objective / d logp(a)
            onehot = np.zeros_like(pi); onehot[np.arange(B), a] = 1.0
            d_obj = g_logp[:, None] * (onehot - pi)              # d objective / d logits

            # Entropy bonus: H = -sum pi log pi ; dH/dz_j = -pi_j (log pi_j + H)
            H = -(pi * logp_all).sum(1)
            d_H = -pi * (logp_all + H[:, None])

            # KL to the reference (playbook clone): KL = sum pi (log pi - log pi_ref)
            ref_logp = log_softmax(ref(o))
            kl = (pi * (logp_all - ref_logp)).sum(1)
            d_kl = pi * ((logp_all - ref_logp) - kl[:, None])

            # Minimize: -objective - ent_coef*H + kl_coef*KL
            d_logits = (-d_obj - cfg.ent_coef * d_H + cfg.kl_coef * d_kl) / B
            actor.adam(actor.backward(cache, d_logits), lr=cfg.lr_actor)

            # ---- value (critic): L_VF = 1/2 (V(s) - R)^2, the loss on the slide ----
            v, vcache = critic.forward(o)
            err = v[:, 0] - R
            critic.adam(critic.backward(vcache, (cfg.vf_coef * err / B)[:, None]), lr=cfg.lr_critic)

            stats["clip_frac"].append(float((~use_unclipped).mean()))
            stats["approx_kl"].append(float((old_logp - logp).mean()))
            stats["entropy"].append(float(H.mean()))
            stats["kl_ref"].append(float(kl.mean()))
            stats["value_loss"].append(float(0.5 * (err ** 2).mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}


# ---------------------------------------------------------------------------
# Evaluation: run a policy on fresh deals and keep score per deal
# ---------------------------------------------------------------------------
def evaluate(policy, n_deals: int = 4000, seed: int = 123, parallel: int = 500):
    """policy(obs) -> actions. Returns per-deal outcomes as a dict of arrays."""
    env = PipelineEnv(parallel, seed=seed)
    obs = env.reset()
    cost = np.zeros(parallel)
    used = np.zeros((parallel, N_ACTIONS))
    start_fit = env.fit.copy()
    out = {k: [] for k in ["revenue", "cost", "won", "weeks", "disqualified", "discounted", "exec_briefings", "fit", "net"]}
    while len(out["net"]) < n_deals:
        a = policy(obs)
        disc_before = np.maximum(env.disc, (a == 4).astype(float))
        obs, _, d, info = env.step(a)
        cost += info["cost"]
        used[np.arange(parallel), a] += 1
        for i in np.where(d)[0]:
            out["revenue"].append(info["revenue"][i]); out["cost"].append(cost[i]); out["won"].append(info["won"][i])
            out["weeks"].append(info["weeks"][i]); out["disqualified"].append(info["disqualified"][i])
            out["discounted"].append(bool(disc_before[i]) and info["won"][i]); out["exec_briefings"].append(used[i, 3])
            out["fit"].append(start_fit[i]); out["net"].append(info["revenue"][i] - cost[i])
            cost[i] = 0; used[i] = 0; start_fit[i] = env.fit[i]
    return {k: np.array(v[:n_deals]) for k, v in out.items()}


def summarize(res: dict) -> dict:
    won = res["won"]
    return {
        "net_per_deal": res["net"].mean(),
        "win_rate": won.mean(),
        "revenue_per_deal": res["revenue"].mean(),
        "cost_per_deal": res["cost"].mean(),
        "weeks_to_close": res["weeks"][won].mean() if won.any() else np.nan,
        "discount_rate_on_wins": res["discounted"][won].mean() if won.any() else np.nan,
        "exec_briefings_per_deal": res["exec_briefings"].mean(),
        "disqualify_rate": res["disqualified"].mean(),
    }


def greedy(actor: MLP):
    return lambda o: actor(o).argmax(1)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------
def train(cfg: PPOConfig = PPOConfig(), verbose: bool = True, eval_every: int = 10):
    rng = np.random.default_rng(cfg.seed)
    actor = MLP(OBS_DIM, N_ACTIONS, cfg.hidden, seed=cfg.seed, out_scale=0.01)
    critic = MLP(OBS_DIM, 1, cfg.hidden, seed=cfg.seed + 1, out_scale=1.0)
    bc_acc = behavior_clone(actor, seed=cfg.seed + 7)
    ref = actor.copy()                                   # frozen reference policy, like the SFT model in RLHF
    env = PipelineEnv(cfg.n_deals, seed=cfg.seed + 11)
    obs = env.reset()
    hist = []
    for it in range(1, cfg.iterations + 1):
        buf, last_val, obs = rollout(env, obs, actor, critic, cfg.horizon, rng)
        adv, ret = gae(buf["rew"], buf["val"], buf["done"], last_val, cfg.gamma, cfg.lam)
        batch = {"obs": buf["obs"].reshape(-1, OBS_DIM), "act": buf["act"].reshape(-1),
                 "logp": buf["logp"].reshape(-1), "adv": adv.reshape(-1), "ret": ret.reshape(-1)}
        ev = 1 - np.var(ret - buf["val"]) / (np.var(ret) + 1e-8)
        stats = ppo_update(actor, critic, ref, batch, cfg, rng)
        row = {"iteration": it, "explained_variance": float(ev), **stats}
        if it % eval_every == 0 or it == 1:
            row["eval_net_per_deal"] = float(evaluate(greedy(actor), n_deals=3000, seed=999)["net"].mean())
            if verbose:
                print(f"iter {it:4d}  net/deal ${row['eval_net_per_deal']:6.2f}K  value loss {stats['value_loss']:.4f}  "
                      f"KL to playbook {stats['kl_ref']:.3f}  clip {stats['clip_frac']:.2f}  EV {ev:.2f}")
        hist.append(row)
    return actor, critic, ref, hist, bc_acc
