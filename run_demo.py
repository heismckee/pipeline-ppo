"""Train a PPO agent on a simulated B2B pipeline and compare it with the current playbook.

Run:  python run_demo.py            (about 2 minutes on a laptop CPU)
Writes charts, CSVs, trained weights and results.md to ./outputs
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_ppo import (LAST0, ACTIONS, N_ACTIONS, OBS_DIM, STAGES, PipelineEnv, PPOConfig, evaluate, greedy,
                          log_softmax, playbook_action, summarize, train)

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
PAPER, INK, RED, RIVER, BRICK, CONCRETE, GRAPHITE = "#F3EEE4", "#121212", "#D02418", "#1F3A4D", "#8E3B26", "#8D8A83", "#4A4741"
plt.rcParams.update({"figure.dpi": 130, "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": GRAPHITE, "ytick.color": GRAPHITE, "font.size": 10})
GAMMA = 0.99


def action_mix_by_stage(policy, n_steps: int = 60, seed: int = 5):
    """Share of each play chosen at each stage, over the states the policy actually visits."""
    env = PipelineEnv(1000, seed=seed)
    obs = env.reset()
    counts = np.zeros((len(STAGES), N_ACTIONS))
    for _ in range(n_steps):
        a = policy(obs)
        st = obs[:, :5].argmax(1)
        np.add.at(counts, (st, a), 1)
        obs, *_ = env.step(a)
    return counts / counts.sum(1, keepdims=True)


def probe_state(stage: int, fit: float, eng: float, week: float = 0.3, last: int = 0):
    o = np.zeros(OBS_DIM)
    o[stage] = 1; o[5] = fit; o[6] = eng; o[7] = week; o[LAST0 + last] = 1
    return o


def snapshot_values(actor, critic, n_steps: int = 220, parallel: int = 1500, seed: int = 77):
    """Run the stochastic policy, record V(s) at every weekly snapshot and the discounted net revenue that followed."""
    env = PipelineEnv(parallel, seed=seed)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    rec_v, rec_stage, rec_r, rec_d, rec_slot = [], [], [], [], []
    for _ in range(n_steps):
        p = np.exp(log_softmax(actor(obs)))
        a = (p.cumsum(1) > rng.random((parallel, 1))).argmax(1)
        rec_v.append(critic(obs)[:, 0] * env.scale); rec_stage.append(obs[:, :5].argmax(1))
        obs, r, d, _ = env.step(a)
        rec_r.append(r * env.scale); rec_d.append(d)
    V, S, R, D = map(np.array, (rec_v, rec_stage, rec_r, rec_d))
    T = len(R)
    G = np.zeros_like(R); complete = np.zeros_like(D, dtype=bool)
    g = np.zeros(parallel); ended = np.zeros(parallel, dtype=bool)
    for t in reversed(range(T)):
        g = R[t] + GAMMA * g * (1 - D[t])
        ended = D[t] | ended                                  # the deal seen at week t finishes at or after t
        G[t] = g; complete[t] = ended
    keep = complete                                           # only snapshots whose deal finished inside the window
    return V[keep], S[keep], G[keep]


def main():
    cfg = PPOConfig(iterations=300, lr_actor=1e-3)
    actor, critic, ref, hist, bc_acc = train(cfg, eval_every=10)
    np.savez(OUT / "trained_agent.npz", **{f"actor_{k}": v for k, v in actor.p.items()},
             **{f"critic_{k}": v for k, v in critic.p.items()})
    hist = pd.DataFrame(hist)
    hist.to_csv(OUT / "training_history.csv", index=False)

    # ---- head-to-head evaluation on the same 8,000 fresh deals ----
    res_pb = evaluate(playbook_action, 8000, seed=2026)
    res_ppo = evaluate(greedy(actor), 8000, seed=2026)
    s_pb, s_ppo = summarize(res_pb), summarize(res_ppo)
    comp = pd.DataFrame({"playbook": s_pb, "ppo": s_ppo})
    comp.to_csv(OUT / "policy_comparison.csv")
    lift = s_ppo["net_per_deal"] / s_pb["net_per_deal"] - 1
    # bootstrap interval on the difference in net value per deal
    rng = np.random.default_rng(0)
    diffs = [rng.choice(res_ppo["net"], 8000).mean() - rng.choice(res_pb["net"], 8000).mean() for _ in range(1000)]
    lo, hi = np.percentile(diffs, [5, 95])

    # ---- 1. training curve ----
    ev = hist.dropna(subset=["eval_net_per_deal"])
    pb_line = evaluate(playbook_action, 3000, seed=999)["net"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ev["iteration"], ev["eval_net_per_deal"], color=RED, lw=2, label="PPO agent")
    ax.axhline(pb_line, color=INK, lw=1.5, ls="--", label="Current playbook")
    ax.set_xlabel("PPO iteration"); ax.set_ylabel("Net value per deal ($K)")
    ax.set_title("The agent starts as a copy of the playbook, then learns past it", loc="left", fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "01_training_curve.png"); plt.close(fig)

    # ---- 2. which play at which stage ----
    mix_pb, mix_ppo = action_mix_by_stage(playbook_action), action_mix_by_stage(greedy(actor))
    pd.DataFrame(mix_ppo, index=STAGES, columns=ACTIONS).round(3).to_csv(OUT / "ppo_action_mix_by_stage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, mix, title in [(axes[0], mix_pb, "Current playbook"), (axes[1], mix_ppo, "PPO agent")]:
        ax.imshow(mix, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(N_ACTIONS)); ax.set_xticklabels(ACTIONS, rotation=30, ha="right")
        ax.set_yticks(range(len(STAGES))); ax.set_yticklabels(STAGES); ax.set_title(title, loc="left", fontweight="bold")
        for i in range(len(STAGES)):
            for j in range(N_ACTIONS):
                if mix[i, j] >= 0.05:
                    ax.text(j, i, f"{mix[i, j]:.0%}", ha="center", va="center", fontsize=8, color=PAPER if mix[i, j] > 0.55 else INK)
    fig.suptitle("Which play the policy runs at each stage", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(OUT / "02_policy_by_stage.png"); plt.close(fig)

    # ---- 3. what the agent learned about exec briefings and discounts ----
    fits = np.linspace(0, 1, 41); engs = np.linspace(0, 1, 41)
    p_exec = np.exp(log_softmax(actor(np.array([probe_state(3, f, 0.5) for f in fits]))))[:, 3]
    p_disc = np.exp(log_softmax(actor(np.array([probe_state(4, 0.45, e) for e in engs]))))[:, 4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(fits, p_exec, color=RED, lw=2); axes[0].axvline(0.6, color=CONCRETE, ls=":", lw=1)
    axes[0].set_title("Exec briefing at Proposal, by ICP fit", loc="left", fontweight="bold")
    axes[0].set_xlabel("ICP fit"); axes[0].set_ylabel("Probability of choosing it"); axes[0].set_ylim(0, 1)
    axes[0].text(0.62, 0.9, "simulator: briefings only\nhelp when fit > 0.6", fontsize=8, color=GRAPHITE, va="top")
    axes[1].plot(engs, p_disc, color=RIVER, lw=2); axes[1].axvline(0.5, color=CONCRETE, ls=":", lw=1)
    axes[1].set_title("Discount at Negotiation, by engagement", loc="left", fontweight="bold")
    axes[1].set_xlabel("Buyer engagement"); axes[1].set_ylim(0, 1)
    axes[1].text(0.52, 0.9, "simulator: discounts only\nrescue cold deals (< 0.5)", fontsize=8, color=GRAPHITE, va="top")
    fig.tight_layout(); fig.savefig(OUT / "03_learned_rules.png"); plt.close(fig)

    # ---- 4. value head as opportunity valuation ----
    V, S, G = snapshot_values(actor, critic)
    stage_mean = pd.Series(G).groupby(S).mean()
    static = stage_mean.reindex(S).to_numpy()
    def r2(pred):
        return 1 - ((G - pred) ** 2).mean() / G.var()
    r2_value, r2_stage = r2(V), r2(static)
    bins = pd.qcut(V, 10, duplicates="drop")
    cal = pd.DataFrame({"pred": V, "actual": G}).groupby(bins, observed=True).mean()
    fig, ax = plt.subplots(figsize=(5.6, 5))
    lim = [min(cal.min()) - 2, max(cal.max()) + 2]
    ax.plot(lim, lim, color=CONCRETE, ls="--", lw=1)
    ax.scatter(cal["pred"], cal["actual"], color=RED, s=50, zorder=3)
    ax.set_xlabel("Predicted opportunity value V(s), $K"); ax.set_ylabel("Actual discounted net revenue, $K")
    ax.set_title("The value head is a calibrated\nopportunity forecast", loc="left", fontweight="bold")
    ax.text(0.03, 0.97, f"Snapshot-level R²\nvalue head: {r2_value:.2f}\nstage averages: {r2_stage:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "04_value_calibration.png"); plt.close(fig)

    # ---- 5. opportunity value over the life of three deals ----
    env = PipelineEnv(400, seed=31); obs = env.reset()
    traj = {i: [] for i in range(400)}; outcome = {}
    alive = np.ones(400, dtype=bool)
    for w in range(26):
        v = critic(obs)[:, 0] * env.scale; st = obs[:, :5].argmax(1)
        for i in np.where(alive)[0]:
            traj[i].append((w, v[i], st[i]))
        obs, _, d, info = env.step(greedy(actor)(obs))
        for i in np.where(alive & d)[0]:
            outcome[i] = "won" if info["won"][i] else ("lost" if info["lost"][i] else "stalled")
        alive &= ~d
    picks = {}
    for i, o in outcome.items():
        if o not in picks and len(traj[i]) >= 8:
            picks[o] = i
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"won": RED, "lost": INK, "stalled": CONCRETE}
    for o, i in picks.items():
        w, v, st = zip(*traj[i])
        ax.plot(w, v, color=colors[o], lw=2, marker="o", ms=3, label=f"Deal that {o}")
        ax.annotate(STAGES[st[-1]], (w[-1], v[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, color=colors[o])
    ax.set_xlabel("Weeks open"); ax.set_ylabel("Opportunity value V(s), $K")
    ax.set_title("Opportunity value updates every week, not just at stage changes", loc="left", fontweight="bold")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(OUT / "05_deal_trajectories.png"); plt.close(fig)

    # ---- 6. scorecard ----
    metrics = [("net_per_deal", "Net value per deal ($K)"), ("cost_per_deal", "Cost per deal ($K)"),
               ("win_rate", "Win rate"), ("discount_rate_on_wins", "Wins that needed a discount"),
               ("exec_briefings_per_deal", "Exec briefings per deal")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.2))
    for ax, (k, label) in zip(axes, metrics):
        vals = [s_pb[k], s_ppo[k]]
        ax.bar(["Playbook", "PPO"], vals, color=[CONCRETE, RED], width=0.6)
        for x, v in enumerate(vals):
            ax.text(x, v, f"{v:.0%}" if k in ("win_rate", "discount_rate_on_wins") else f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(label, loc="left", fontsize=9, fontweight="bold"); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT / "06_scorecard.png"); plt.close(fig)

    md = f"""# Results

Trained for {cfg.iterations} PPO iterations ({cfg.iterations * cfg.n_deals * cfg.horizon:,} simulated deal-weeks).
Behavior cloning matched the playbook on {bc_acc:.0%} of states before PPO started.

| Metric | Current playbook | PPO agent |
|---|---|---|
| Net value per deal | ${s_pb['net_per_deal']:.2f}K | **${s_ppo['net_per_deal']:.2f}K** |
| Win rate | {s_pb['win_rate']:.1%} | {s_ppo['win_rate']:.1%} |
| Revenue per deal | ${s_pb['revenue_per_deal']:.2f}K | ${s_ppo['revenue_per_deal']:.2f}K |
| Cost per deal | ${s_pb['cost_per_deal']:.2f}K | ${s_ppo['cost_per_deal']:.2f}K |
| Wins that needed a discount | {s_pb['discount_rate_on_wins']:.0%} | {s_ppo['discount_rate_on_wins']:.0%} |
| Exec briefings per deal | {s_pb['exec_briefings_per_deal']:.2f} | {s_ppo['exec_briefings_per_deal']:.2f} |
| Weeks to close (wins) | {s_pb['weeks_to_close']:.1f} | {s_ppo['weeks_to_close']:.1f} |

Net value per deal improved **{lift:+.1%}** (90% bootstrap interval on the difference: ${lo:.2f}K to ${hi:.2f}K per deal), measured on the same 8,000 fresh simulated deals.

Value head as an opportunity forecast: snapshot-level R² **{r2_value:.2f}** vs **{r2_stage:.2f}** for stage averages.
"""
    (OUT / "results.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
