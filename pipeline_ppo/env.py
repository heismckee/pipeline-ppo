"""A simulated B2B sales pipeline where each episode is one opportunity.

Every week the agent picks one play for each open deal. The deal's engagement,
stage and chance of closing respond to that play. Revenue arrives only when a
deal closes, which is exactly the "one reward at the end" shape of RLHF.

The simulator is vectorized: it runs N deals in parallel, and finished deals are
replaced with new ones automatically.
"""
from __future__ import annotations

import numpy as np

STAGES = ["Discovery", "Qualified", "Solution", "Proposal", "Negotiation"]
ACTIONS = ["Nurture email", "SDR call", "Case study", "Exec briefing", "Discount 15%", "Disqualify"]
N_STAGES, N_ACTIONS = len(STAGES), len(ACTIONS)

# Cost of each play in $K of team time and money.
ACTION_COST = np.array([0.05, 0.4, 0.6, 3.0, 0.0, 0.0])
HOLDING_COST = 0.15        # $K per open deal per week (rep attention, forecasting overhead)
DISCOUNT = 0.15            # revenue given up when a discount is offered
SPONSOR_UPLIFT = 0.25      # scope expansion when an executive sponsor is secured
MAX_WEEKS = 26             # deals that haven't closed in two quarters are counted as stalled

# Engagement lift from each play at each stage (rows = actions, cols = stages).
# These are the hidden "physics" of the market. The agent never sees this table.
ENGAGE = np.array([
    # Disc  Qual  Sol   Prop  Neg
    [0.03, 0.03, 0.02, 0.01, 0.01],   # Nurture email
    [0.22, 0.18, 0.06, 0.03, 0.02],   # SDR call
    [0.05, 0.14, 0.24, 0.10, 0.04],   # Case study
    [0.02, 0.04, 0.08, 0.10, 0.10],   # Exec briefing (big effect comes through ADVANCE, for high-fit deals)
    [0.00, 0.00, 0.02, 0.04, 0.06],   # Discount
    [0.00, 0.00, 0.00, 0.00, 0.00],   # Disqualify
])
# Direct push on the chance of advancing a stage (logit units).
ADVANCE = np.array([
    [0.00, 0.00, 0.00, 0.00, 0.00],
    [0.35, 0.30, 0.00, 0.00, 0.00],
    [0.00, 0.20, 0.45, 0.10, 0.00],
    [0.00, 0.00, 0.10, 0.20, 0.20],   # plus a fit-dependent bonus below
    [0.00, 0.00, 0.00, 0.30, 0.70],   # plus a low-engagement bonus at Negotiation
    [0.00, 0.00, 0.00, 0.00, 0.00],
])

LAST0 = 11                       # index where the last-play one-hot starts
OBS_DIM = LAST0 + N_ACTIONS


class PipelineEnv:
    """Vectorized pipeline simulator.

    Observation per deal (what a rep or CRM would know):
      stage one-hot (5), fit, engagement, weeks open / 26, log deal size (scaled),
      discount already given (0/1), executive sponsor secured (0/1), last play one-hot (6).
    Reward: closed revenue minus the cost of plays and holding cost, in $K / 100.
    """

    def __init__(self, n_deals: int = 256, seed: int = 0, reward_scale: float = 100.0):
        self.n = n_deals
        self.rng = np.random.default_rng(seed)
        self.scale = reward_scale
        self.stage = np.zeros(n_deals, dtype=int)
        self.fit = np.zeros(n_deals)
        self.eng = np.zeros(n_deals)
        self.week = np.zeros(n_deals, dtype=int)
        self.size = np.zeros(n_deals)
        self.disc = np.zeros(n_deals)
        self.sponsor = np.zeros(n_deals)
        self.last = np.zeros(n_deals, dtype=int)
        self._new(np.arange(n_deals))

    # ---- deal generation -------------------------------------------------
    def _new(self, idx: np.ndarray) -> None:
        k = len(idx)
        if k == 0:
            return
        self.stage[idx] = 0
        self.fit[idx] = self.rng.beta(2.0, 2.0, k)                     # ICP fit score, 0..1
        self.eng[idx] = self.rng.uniform(0.05, 0.35, k)                # starting engagement
        self.week[idx] = 0
        self.size[idx] = np.exp(self.rng.normal(np.log(55), 0.45, k))  # deal size in $K, median ~55K
        self.disc[idx] = 0.0
        self.sponsor[idx] = 0.0
        self.last[idx] = 0

    def reset(self) -> np.ndarray:
        self._new(np.arange(self.n))
        return self.obs()

    def obs(self) -> np.ndarray:
        o = np.zeros((self.n, OBS_DIM))
        o[np.arange(self.n), self.stage] = 1.0
        o[:, 5] = self.fit
        o[:, 6] = self.eng
        o[:, 7] = self.week / MAX_WEEKS
        o[:, 8] = (np.log(self.size) - np.log(55)) / 0.45
        o[:, 9] = self.disc
        o[:, 10] = self.sponsor
        o[np.arange(self.n), LAST0 + self.last] = 1.0
        return o

    # ---- one week ---------------------------------------------------------
    def step(self, a: np.ndarray):
        n, r = self.n, self.rng
        s = self.stage.copy()
        reward = -ACTION_COST[a] - HOLDING_COST
        done = np.zeros(n, dtype=bool)
        won = np.zeros(n, dtype=bool)

        # Disqualify ends the deal immediately with no revenue.
        dq = a == 5
        done |= dq

        # Engagement: repeated plays fatigue the buyer, so the same play twice in a row works half as well.
        fatigue = np.where(a == self.last, 0.5, 1.0)
        lift = ENGAGE[a, s] * fatigue * (0.6 + 0.8 * self.fit)
        self.eng = np.clip(self.eng + lift * (1 - self.eng) - 0.04 * self.eng, 0, 1)

        # A discount anywhere sets the flag; the revenue cut applies at close.
        self.disc = np.where(a == 4, 1.0, self.disc)

        # Chance of advancing a stage this week.
        logit = -2.6 + 2.2 * self.fit + 2.6 * self.eng + ADVANCE[a, s] * fatigue
        exec_bonus = (a == 3) * (s >= 3) * np.where(self.fit > 0.6, 0.6, -0.3)   # executives only help strong-fit late deals
        disc_bonus = (a == 4) * (s == 4) * np.where(self.eng < 0.5, 0.9, 0.0)     # discounts only rescue cold negotiations
        # An exec briefing on a strong-fit, late-stage deal wins an executive sponsor, who expands the scope of the deal.
        self.sponsor = np.where((a == 3) & (s >= 3) & (self.fit > 0.6), 1.0, self.sponsor)
        logit = logit + exec_bonus + disc_bonus
        p_adv = 1 / (1 + np.exp(-logit))
        p_lost = 1 / (1 + np.exp(-(-4.2 + 2.2 * (1 - self.eng) + 1.8 * (1 - self.fit) + 0.06 * self.week)))

        u = r.random(n)
        lost = (u < p_lost) & ~dq
        adv = (~lost) & (r.random(n) < p_adv) & ~dq

        close = adv & (s == 4)
        won |= close
        value = self.size * (1 + SPONSOR_UPLIFT * self.sponsor) * (1 - DISCOUNT * self.disc)
        reward = reward + np.where(close, value, 0.0)
        self.stage = np.where(adv & (s < 4), s + 1, s)

        done |= lost | close
        self.week += 1
        stalled = (self.week >= MAX_WEEKS) & ~done
        done |= stalled
        self.last = a.copy()

        info = {"won": won, "lost": lost, "disqualified": dq, "stalled": stalled,
                "revenue": np.where(close, value, 0.0), "sponsor": self.sponsor.copy(),
                "cost": ACTION_COST[a] + HOLDING_COST, "weeks": self.week.copy(), "fit": self.fit.copy()}
        # Replace finished deals with new ones so every slot keeps producing experience.
        self._new(np.where(done)[0])
        return self.obs(), reward / self.scale, done, info


def playbook_action(obs: np.ndarray) -> np.ndarray:
    """The 'current playbook': a sensible human rule set, one play per stage.

    Discovery/Qualified: SDR call. Solution: case study. Proposal: exec briefing.
    Negotiation: discount. Never disqualify. Alternates with a nurture email when
    the same play was just used, to avoid obvious fatigue.
    """
    stage = obs[:, :N_STAGES].argmax(1)
    last = obs[:, LAST0:].argmax(1)
    play = np.array([1, 1, 2, 3, 4])[stage]
    return np.where(play == last, 0, play)
