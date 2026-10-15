# PPO and RLHF: Reference Sheet

Companion to the *Pipeline PPO* code-along. Every equation here appears in the code.

## The loop in five steps

1. **Clone a reference policy.** Imitate the playbook (RLHF: supervised fine-tuning). Freeze a copy as π_ref.
2. **Collect experience.** Run the current policy on many deals for a few weeks. Store state, action, log-probability, value and reward.
3. **Score the experience.** Compute advantages with GAE: which plays did better than the forecast expected?
4. **Update.** Several passes of the clipped policy objective, the value loss, an entropy bonus and the KL penalty.
5. **Repeat.** The new policy collects the next batch.

## Symbols

| Symbol | Meaning | In this project |
|---|---|---|
| s, s_t | State | Deal snapshot: stage, fit, engagement, weeks open, size, discount and sponsor flags, last play |
| a, a_t | Action | One of six plays |
| r_t | Reward | Revenue at close minus play and holding costs, in $100K |
| π_θ(a\|s) | Policy (actor) | Probability of each play, from a softmax over logits |
| π_old | Policy that collected the batch | Frozen during the update |
| π_ref | Reference policy | The playbook clone (RLHF: the SFT model) |
| V_θ(s) | Value function (critic) | Opportunity value: expected discounted net revenue from s |
| γ | Discount factor | 0.99 per week |
| λ | GAE parameter | 0.95 |
| ε | Clip range | 0.2 |
| β | KL coefficient | 0.02 |
| c_ent | Entropy coefficient | 0.01 |

## Core equations

**Discounted return** (what actually happened from state s)

  G_t = r_t + γ r_{t+1} + γ² r_{t+2} + … = Σ_k γ^k r_{t+k}

**Value loss** (trains the critic; the equation from the lesson)

  L^VF = ½ (V_θ(s_t) − R_t)²   where the target R_t = A_t + V_old(s_t)

**One-step surprise (TD error)**

  δ_t = r_t + γ V(s_{t+1}) (1 − done_t) − V(s_t)

**Generalized advantage estimation**

  A_t = δ_t + γ λ (1 − done_t) A_{t+1}
  λ = 1 gives the full Monte Carlo return minus V (unbiased, noisy). λ = 0 gives δ_t alone (stable, biased).

**Probability ratio**

  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t) = exp(log π_θ − log π_old)

**Clipped policy objective** (maximize)

  L^CLIP = E[ min( r_t A_t,  clip(r_t, 1−ε, 1+ε) A_t ) ]

**Entropy** (exploration bonus, maximize)

  H(π) = −Σ_a π(a|s) log π(a|s)

**KL to the reference** (penalty, minimize)

  KL(π_θ ‖ π_ref) = Σ_a π_θ(a|s) [log π_θ(a|s) − log π_ref(a|s)]

**Total loss minimized each minibatch**

  L = −L^CLIP + c_vf · L^VF − c_ent · H + β · KL

## Gradients written out in the code (softmax policy, logits z)

| Quantity | Gradient with respect to logit z_j |
|---|---|
| log π(a) | 1[j = a] − π_j |
| Clipped objective | g · (1[j = a] − π_j), where g = r·A if the unclipped term is the minimum, else 0 |
| Entropy H | −π_j (log π_j + H) |
| KL to reference | π_j [(log π_j − log π_ref,j) − KL] |
| Cross-entropy (cloning) | π_j − target_j |
| Value loss | V − R, with respect to the critic's output |

## Reading the training log

| Signal | Healthy range | What it tells you |
|---|---|---|
| Net value per deal | Rises above the playbook, then plateaus | The thing you care about |
| Value loss | Falls, then noisy-flat | How well the critic forecasts |
| Explained variance | 0.4 to 0.9 here | Share of return variation the critic explains (1 = perfect, 0 = useless) |
| Clip fraction | 1% to 20% | How often the clip limited the update. Very high means the learning rate is too aggressive |
| Approximate KL (old vs new) | Below about 0.02 per update | Size of each policy step |
| KL to reference | Rises, then levels off | Distance from the playbook. Unbounded growth means the leash is too loose |
| Entropy | Declines slowly | Exploration. A collapse to near zero means the policy stopped trying alternatives |

## Hyperparameters and what moving them does

| Knob | Raise it | Lower it |
|---|---|---|
| γ (discount) | Values later revenue more; longer credit assignment | More short-term; less noise |
| λ (GAE) | Trusts real rewards more; less bias, more variance | Trusts the critic more; more bias, less variance |
| ε (clip) | Bigger steps per update; faster but riskier | Smaller, safer steps |
| β (KL) | Stays closer to the playbook | Freer to deviate (and, in RLHF, to exploit reward-model flaws) |
| c_ent | More exploration | Faster commitment, risk of premature collapse |
| Epochs per batch | Squeezes more from each batch; risk of overfitting it | More stable, needs more data |
| Learning rate | Faster learning, more instability | Slower, steadier |

## RLHF mapping

| RLHF | Pipeline PPO |
|---|---|
| Prompt + tokens so far | Deal state |
| Next token | Next play |
| Completion | One deal from open to close |
| SFT model / reference | Playbook clone |
| Reward model | Revenue minus cost (here the true reward; in RLHF a learned and imperfect proxy) |
| Value head on the LLM | Opportunity value |
| KL penalty per token | KL penalty per weekly decision |
| Reward hacking | Exploiting a simulator quirk that wouldn't hold in the real market |

## Results from the reference run

Net value per deal $15.65K (playbook) → $18.22K (PPO), +16.4%. Discounted wins 100% → 9%. Value-head R² 0.44 vs 0.11 for stage averages. KL experiment (120 iterations): β = 0 → $17.39K, KL 0.51; β = 0.02 → $16.54K, KL 0.30; β = 0.2 → $14.50K, KL 0.05.
