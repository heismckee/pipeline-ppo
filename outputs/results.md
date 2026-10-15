# Results

Trained for 300 PPO iterations (2,457,600 simulated deal-weeks).
Behavior cloning matched the playbook on 100% of states before PPO started.

| Metric | Current playbook | PPO agent |
|---|---|---|
| Net value per deal | $15.65K | **$18.22K** |
| Win rate | 31.9% | 31.4% |
| Revenue per deal | $19.14K | $21.87K |
| Cost per deal | $3.48K | $3.65K |
| Wins that needed a discount | 100% | 9% |
| Exec briefings per deal | 0.51 | 0.50 |
| Weeks to close (wins) | 7.6 | 7.8 |

Net value per deal improved **+16.4%** (90% bootstrap interval on the difference: $1.65K to $3.46K per deal), measured on the same 8,000 fresh simulated deals.

Value head as an opportunity forecast: snapshot-level R² **0.44** vs **0.11** for stage averages.
