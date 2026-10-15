# Video Script: Teaching a Sales Pipeline to Learn

**Format:** screen recording of the Colab notebook, with face cam in the corner. About 16 minutes.
**On screen** notes say what to show. **Say** is the narration. Times are targets, not rules.

---

## 0:00 Cold open

**On screen:** Results table from the README: playbook $15.65K per deal, PPO $18.22K.

**Say:**
This agent took a reasonable sales playbook and improved net revenue per deal by 16%. It did it without winning more deals. It stopped discounting deals that were going to close anyway, and it started saving executive time for the deals where it changes the outcome.

It's trained with PPO, the same algorithm used to train chat models with human feedback. I'm learning RLHF right now, and the fastest way I know to understand an algorithm is to point it at a problem I've worked on for fifteen years. So that's what we're doing.

## 0:45 The setup

**On screen:** The RLHF mapping table at the top of the notebook.

**Say:**
In RLHF, a language model writes one token at a time, and at the end a reward model scores the whole answer. A sales deal works the same way. Each week you pick one play: a nurture email, an SDR call, a case study, an exec briefing, a discount, or you disqualify it. Revenue shows up once, at the end, if the deal closes.

Every part of RLHF has a pipeline twin. The prompt is the deal's current state. The next token is the next play. The supervised model is a copy of our current playbook. And the value head, which is the part I want you to watch, becomes an opportunity forecast.

## 1:45 Part 1: the simulator

**On screen:** The hidden-rules list, then the one-deal printout.

**Say:**
I can't run a million experiments on real enterprise deals, so this is a simulated market. It has rules the agent never sees. Exec briefings only pay off on strong-fit deals late in the cycle, where they win an executive sponsor and expand the contract. Discounts only rescue negotiations that have gone cold. Repeating the same play two weeks running works half as well.

Here's one deal under the current playbook. Notice the reward is negative almost every week. That's the cost of plays and rep time. Then one big number when it closes. That's the hard part of this problem: which of those nine weeks earned the win?

## 3:00 Part 2: the baseline

**On screen:** The baseline table.

**Say:**
The playbook isn't a straw man. It beats random plays by a lot. But look at two numbers: it discounts every single win, and it sends an exec briefing on about half of all deals. Hold onto those.

## 3:40 Part 3: the network, by hand

**On screen:** The `MLP` class, then the gradient check output.

**Say:**
No PyTorch here. This is a small neural network in plain NumPy, and `backward` is the chain rule written out line by line. Whenever you write your own backprop, you check it. Nudge one weight, measure what happens, and compare with what the code claims. They match to six decimal places, so we can trust it.

## 4:40 Part 4: clone the playbook

**On screen:** `behavior_clone` and the 100% match.

**Say:**
RLHF never starts from a random model. It starts from a model that already behaves well, then keeps a frozen copy as a reference. We do the same thing: train the network to imitate the playbook. It matches on every state. That copy becomes the reference the agent gets measured against for the rest of training.

## 5:30 Part 5: the value loss

**On screen:** The equation, then the worked example cell.

**Say:**
This is the equation that started this project. The value function, V of s, is a forecast: from this deal's current state, how much net revenue do we expect? The value loss is just the squared gap between that forecast and what actually happened, cut in half.

Worked example: the forecast says $34K. The deal closes eight weeks later for $187K. Discounted, that's about $168K. The loss is big, and the advantage, actual minus forecast, is plus $134K. That deal beat expectations.

**On screen:** The GAE cell.

**Say:**
PPO doesn't wait for every deal to finish. Generalized advantage estimation blends what actually happened with what the forecast now expects. Lambda sets the blend. At one, it's the full sum from the slide. At zero, it trusts the forecast after one week.

## 7:30 Part 6: PPO's clip

**On screen:** The clipping plot.

**Say:**
Here's the core of PPO. For every play the agent ran, compare how likely that play is now with how likely it was when we collected the data. If the play beat the forecast, its probability can go up, but only by about 20% per update. If it lost, it can go down by about 20%. The flat parts of the red line are where the update stops. One lucky batch of deals can't rewrite the playbook.

Two more terms come straight from RLHF. An entropy bonus keeps the agent exploring. And a KL penalty charges the agent for every move away from the playbook copy. That's the leash.

## 9:00 Part 7: train

**On screen:** Run the training cell; talk over the log.

**Say:**
Two minutes on a free Colab CPU. Watch four numbers. Net value per deal climbs past the playbook early. Value loss settles. KL to the playbook rises and then levels off, which is the leash working. Explained variance gets to about a half: the forecast explains half of what happens to a deal, which is good for something as noisy as a single opportunity.

## 10:30 Part 8: what it learned

**On screen:** Comparison table, then the heatmap.

**Say:**
Net value per deal is up 16%. Win rate is basically flat. So where did the money come from?

The heatmap shows it. Early stages barely changed, because the playbook was already right there and the leash charges for every change. The changes are all late-stage. Discounting dropped from every win to about one in ten.

**On screen:** The two probe curves.

**Say:**
This is my favorite chart. I held everything constant and varied one thing. The dotted lines are the simulator's hidden rules. The agent was never told them. The exec-briefing curve climbs right where fit crosses 0.6. The discount curve falls right where engagement crosses 0.5. It found both rules from outcomes alone.

## 12:30 Part 9: the forecast you can use Monday

**On screen:** R² printout and calibration chart.

**Say:**
Even if you never let an agent choose plays, this part works on real data today. The value head, trained only with that value loss, is a dynamic opportunity value. It explains 0.44 of the variation in what deals return, versus 0.11 for the classic stage-percentage forecast. And it's calibrated: deals it values at $30K return about $30K on average.

## 13:45 Part 10: the leash experiment

**On screen:** Experiment A table.

**Say:**
Last experiment. With no KL penalty, the agent earns the most here, $17.39K per deal, but it wanders the furthest from the playbook. With a heavy penalty, it barely moves and keeps discounting everything.

In this simulator the reward is the truth, so the loose leash wins. In RLHF, the reward model is a learned approximation, and a model with no leash finds ways to score well that aren't actually good. In a sales org, the simulator is the approximation. That's why I'd keep the leash.

## 14:45 What transfers

**On screen:** "What transfers to real pipelines" section.

**Say:**
Three things to take back to a real team. First, train a value head on your CRM snapshots. That's a better forecast, and you don't need an agent. Second, PPO learns by trying, so for live deals start with contextual bandits and randomized holdouts. Third, logged CRM data is biased: reps already send exec briefings to deals they think will close. That's the same causal problem as incrementality testing, and it needs the same fix.

## 15:30 Close

**On screen:** Repo README and the Colab badge.

**Say:**
Code, notebook, a reference sheet and a question set are linked below. Run it, break it, change the market and see what the agent learns. If you build the value head on your own pipeline data, I'd like to hear how it compares with your stage forecast.

---

**Description box checklist:** GitHub link, Colab link, chapter timestamps above, the reference sheet PDF, and one line on the course this came from (DeepLearning.AI, *Generative AI with Large Language Models*).
