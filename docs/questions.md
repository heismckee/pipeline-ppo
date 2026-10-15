# Questions for Pipeline PPO

Three sets: **consideration** questions to think through (no single right answer), **evaluation** questions to check your understanding (answer key at the end), and **further study** to take it beyond the code-along.

## Part 1. Questions for consideration

1. The reward is net revenue per deal. What behavior would you expect if it were win rate instead? What if it were revenue with no cost for plays? Which reward would your CFO choose, and which would your sales team choose?
2. The agent learned to stop discounting deals that would close anyway. In a real company, who would push back on that change, and what evidence would convince them?
3. The KL penalty keeps the agent near the playbook. Is that a technical choice, a governance choice, or both? Who should set β in a real sales organization?
4. In RLHF the reward model is learned from human preferences and can be wrong. What is the equivalent risk here if the simulator is calibrated from CRM data? Name one simulator flaw an agent might exploit.
5. The agent never disqualifies deals. Is that a failure of learning, a property of the simulated market, or a sign that disqualifying is rarely worth it? How would you find out?
6. A value head can be trained on historical data without ever letting an agent act. What decisions would you make differently tomorrow with a dynamic opportunity value instead of stage percentages?
7. Reps choose plays based on information the CRM doesn't capture (tone on a call, a champion leaving). How does that hidden information bias a model trained on logged data, and what experiment design would correct it?
8. The policy uses a deal's current state but no memory of earlier weeks beyond the last play. What information is lost, and when would it matter?
9. If the agent recommended an exec briefing your sales VP disagreed with, what should happen? Design the human-in-the-loop approval step.
10. Where else in marketing does the same structure appear: a sequence of decisions, a delayed outcome and a need for a forecast of value along the way?

## Part 2. Evaluation questions

1. In one sentence each, what do the actor and the critic output?
2. A deal's critic predicts 0.34. The discounted return that followed is 1.68. Compute the value loss and the advantage.
3. With γ = 0.9 and a reward of 1.0 arriving three steps from now (and zero before it), what is the discounted return today?
4. What do λ = 1 and λ = 0 each reduce GAE to? Which has more variance?
5. A play had advantage +2 and its probability ratio is now 1.35, with ε = 0.2. Does this sample still push the play's probability up? Why or why not?
6. Same question with advantage −2 and ratio 0.75.
7. Why does RLHF (and this project) start PPO from a cloned policy instead of a random one?
8. What does the KL-to-reference term measure, and what happened in the experiment when β went from 0.02 to 0.2?
9. Training shows clip fraction 0.45 and approximate KL 0.15 per update. What's wrong, and what would you change?
10. Explained variance is −0.5 at iteration 1 and 0.5 by iteration 50. What does that tell you?
11. The agent's win rate is slightly lower than the playbook's, but net value is 16% higher. Explain how.
12. Why is the value head's R² (0.44) so much higher than the stage-average forecast's (0.11)?
13. Why is the entropy bonus needed if the clip already limits how fast the policy changes?
14. Why do we normalize advantages within each minibatch?
15. What does gradient-norm clipping in `adam` protect against?

## Part 3. Further study

**Read**
* Schulman et al., *Proximal Policy Optimization Algorithms* (2017): the original PPO paper.
* Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (2015): the source of GAE.
* Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback* (2022): the InstructGPT paper that made RLHF standard.
* Rafailov et al., *Direct Preference Optimization* (2023): RLHF without a separate reward model or PPO loop.
* Sutton and Barto, *Reinforcement Learning: An Introduction*, chapters 6 (temporal-difference learning) and 13 (policy gradients). Free online.
* Huang et al., *The 37 Implementation Details of Proximal Policy Optimization* (ICLR blog track, 2022).

**Build next**
1. **Value head on real CRM data.** Export weekly opportunity snapshots, train V(s) with the value loss, and compare its calibration and R² with your stage-weighted forecast.
2. **Contextual bandit for next-best-action.** Replace PPO with a one-step bandit (Thompson sampling or LinUCB), run it on the simulator, and compare sample efficiency.
3. **Offline RL.** Generate logged data from the playbook with some randomness, then train with an offline method (conservative Q-learning or implicit Q-learning) and see how far it gets without interacting.
4. **Reward hacking on purpose.** Add a simulator bug (for example, discounts that stack without limit) and train with β = 0. Watch the agent exploit it, then show how the KL leash limits the damage.
5. **Portfolio constraints.** Give the team a weekly budget of exec briefings and add it to the environment. How does the learned policy ration a scarce resource?
6. **Compare with DPO.** Generate preference pairs (deal trajectories ranked by net value) and train a policy with DPO instead of PPO.

---

## Answer key for Part 2

1. The actor outputs a probability for each play given the deal state. The critic outputs one number: the expected discounted net revenue from that state (opportunity value).
2. Value loss = ½(0.34 − 1.68)² = ½(1.7956) ≈ **0.898**. Advantage = 1.68 − 0.34 = **+1.34**: the deal did much better than forecast.
3. 0.9³ × 1.0 = **0.729**.
4. λ = 1 gives the full discounted return minus V(s) (Monte Carlo advantage). λ = 0 gives the one-step TD error δ_t. λ = 1 has more variance; λ = 0 has more bias.
5. **No.** For a positive advantage the objective is capped at (1 + ε)A = 1.2 × 2. The ratio 1.35 is above 1.2, so the clipped term is the minimum and its gradient is zero. The play has already moved as far as one update allows.
6. **No.** For a negative advantage the cap is at 1 − ε = 0.8. The ratio 0.75 is below 0.8, so min(0.75 × −2, 0.8 × −2) = min(−1.5, −1.6) = −1.6, the clipped term, and the gradient is zero.
7. A cloned policy already behaves sensibly, so experience is useful from the first batch, and it provides a reference to measure drift against. A random start wastes most early experience on terrible actions.
8. How far the current policy's action probabilities have moved from the playbook clone. At β = 0.2 the agent stayed almost on the playbook (KL 0.05) and kept discounting 99% of wins, giving up most of the gain ($14.50K vs $16.54K per deal).
9. Updates are too large: the policy changes a lot each step and the clip is constantly active. Lower the learning rate, reduce epochs per batch, or tighten ε.
10. At the start the critic is worse than predicting the average (negative). By iteration 50 it explains half the variance in returns, so advantages are becoming informative.
11. It stopped giving away 15% on deals that would have closed anyway and targeted exec briefings at deals where a sponsor expands the contract by 25%. Revenue per win rose while costs stayed flat. Win rate isn't the objective; net value is.
12. Stage averages use one fact (stage). The value head uses fit, engagement, size, weeks open, flags and the policy that will be followed, so it separates strong and weak deals within the same stage.
13. The clip limits step size, not direction. Without an entropy bonus the policy can still march steadily toward a few plays and stop exploring alternatives that might be better.
14. Advantage scale changes as the critic improves and rewards vary across batches. Normalizing keeps the effective step size stable and centers advantages so roughly half the samples push up and half push down.
15. A rare huge gradient (one surprising deal) taking a damaging step. Clipping the global norm keeps any single update bounded.
