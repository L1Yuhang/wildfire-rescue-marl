# Fire Rescue Multi-Agent Reinforcement Learning

Course project: multi-agent air-ground cooperative search and rescue in a dynamic wildfire environment.

Agents:

- UAV searches from the air and discovers survivors.
- UGV moves on the ground, rescues discovered survivors, and returns to base.

Training uses a centralized DQN over joint actions:

```text
joint_action = uav_action * 5 + ugv_action
```

Use the requested conda environment:

```powershell
conda activate RLearning
```

or:

```powershell
conda run -n RLearning python scripts/check_env.py
```

See `PROJECT_PLAN.md` for the full experiment guide.

## Current Verified Assets

The following Easy workflow has been verified in `RLearning`:

```powershell
conda run -n RLearning python scripts/check_env.py
conda run -n RLearning python scripts/train_dqn.py --config configs/env_easy.yaml --seed 2 --timesteps 60000 --expert-episodes 60
conda run -n RLearning python scripts/eval_agents.py --difficulty easy --episodes 30 --seed 2 --include-rl
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty easy --format gif --seed 2
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty easy --format mp4 --seed 2
conda run -n RLearning python scripts/plot_results.py
```

Generated key files:

```text
outputs/models/dqn/dqn_easy_seed2_best.zip
outputs/videos/dqn_easy_demo.gif
outputs/videos/dqn_easy_demo.mp4
outputs/figures/training_reward_curve.png
outputs/figures/eval_success_rate.png
```

The verified Easy DQN policy completes the full sequence:

```text
UAV search -> survivor discovered -> UGV pickup -> UGV returns to base
```

The Medium experiment has also been completed with a larger 16x16 dynamic-fire map:

```powershell
conda run -n RLearning python scripts/train_dqn.py --config configs/env_medium.yaml --seed 0 --expert-episodes 60
conda run -n RLearning python scripts/eval_agents.py --difficulty medium --episodes 50 --seed 0 --include-rl
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty medium --format gif --seed 0
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty medium --format mp4 --seed 0
```

Medium DQN final evaluation over 50 episodes:

```text
DQN success rate: 0.88
DQN average reward: 279.65
DQN average steps: 71.04
A* success rate: 1.00
A* average steps: 53.00
Greedy success rate: 0.00
Random success rate: 0.00
```

Medium generated files:

```text
outputs/models/dqn/dqn_medium_seed0_best.zip
outputs/models/dqn/dqn_medium_seed0.zip
outputs/videos/dqn_medium_demo.gif
outputs/videos/dqn_medium_demo.mp4
outputs/videos/astar_medium_demo.gif
outputs/eval/metrics_csv/summary_medium_seed0.csv
outputs/figures/eval_success_rate.png
outputs/figures/training_reward_curve.png
```

The Hard experiment has been completed on the 20x20 dynamic-fire map. Hard uses
fixed survivor/fire positions for the main report experiment, while the fully
randomized hard setting is kept separately as a diagnostic challenge in
`configs/env_hard_random.yaml`.

```powershell
conda run -n RLearning python scripts/train_dqn.py --config configs/env_hard.yaml --seed 1 --timesteps 120000 --expert-episodes 120 --bc-episodes 160 --bc-steps 3000
conda run -n RLearning python scripts/eval_agents.py --difficulty hard --episodes 50 --seed 1 --include-rl
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty hard --format gif --seed 1
conda run -n RLearning python scripts/render_demo.py --agent dqn --difficulty hard --format mp4 --seed 1
```

Hard DQN final evaluation over 50 episodes:

```text
DQN success rate: 0.98
DQN average reward: 355.10
DQN average steps: 94.14
A* success rate: 1.00
A* average steps: 75.00
Greedy success rate: 0.00
Random success rate: 0.00
```

Hard generated files:

```text
outputs/models/dqn/dqn_hard_seed1_best.zip
outputs/models/dqn/dqn_hard_seed1.zip
outputs/videos/dqn_hard_seed1_demo.gif
outputs/videos/dqn_hard_seed1_demo.mp4
outputs/videos/astar_hard_demo.gif
outputs/eval/metrics_csv/summary_hard_seed1.csv
outputs/figures/render_review/dqn_hard_seed1_frame_00.png
outputs/figures/render_review/dqn_hard_seed1_frame_01.png
outputs/figures/render_review/dqn_hard_seed1_frame_02.png
```

## Final Multi-UGV Experiment Line

The final project line uses `1 UAV + 2 UGV + multiple survivors` in
`FireRescueMultiUGVEnv`.  The centralized action space is `5^3 = 125`.

Main files:

```text
src/fire_rescue_rl/envs/generated_maps.py
src/fire_rescue_rl/envs/fire_rescue_multi_ugv_env.py
src/fire_rescue_rl/agents/astar_multi_ugv.py
src/fire_rescue_rl/agents/multi_ugv_baselines.py
scripts/train_multi_dqn.py
scripts/eval_multi_dqn.py
scripts/render_multi_policy.py
scripts/plot_multi_all.py
```

The final DQN models are behavior-cloned from A* expert trajectories
(`DQN-BC`).  This is intentional: direct DQN fine-tuning in the 125-action
joint space was observed to degrade the policy, so the report should describe
the final model as an expert-pretrained neural policy, not as pure exploration
from scratch.

Completed final results over 30 episodes:

```text
multi_easy:   DQN-BC success 1.00, reward 415.20, steps 92.00,  delivered 2/2
multi_medium: DQN-BC success 1.00, reward 526.42, steps 86.00,  delivered 3/3
multi_hard:   DQN-BC success 1.00, reward 544.17, steps 104.00, delivered 3/3
```

Key commands:

```powershell
conda run -n RLearning python scripts/train_multi_dqn.py --config configs/env_multi_hard.yaml --seed 0 --timesteps 1 --expert-episodes 0 --bc-episodes 320 --bc-steps 5000 --eval-freq 10000
conda run -n RLearning python scripts/eval_multi_dqn.py --config configs/env_multi_hard.yaml --seed 0 --episodes 30 --include-rl
conda run -n RLearning python scripts/render_multi_policy.py --config configs/env_multi_hard.yaml --agent dqn --seed 0 --format gif --tag multi_dqn_hard_demo
conda run -n RLearning python scripts/plot_multi_all.py
```

Final generated assets:

```text
outputs/models/dqn_multi/dqn_multi_easy_seed0_bc.zip
outputs/models/dqn_multi/dqn_multi_medium_seed0_bc.zip
outputs/models/dqn_multi/dqn_multi_hard_seed0_bc.zip
outputs/videos/multi_dqn_easy_demo.gif
outputs/videos/multi_dqn_medium_demo.gif
outputs/videos/multi_dqn_hard_demo.gif
outputs/videos/multi_dqn_hard_demo.mp4
outputs/figures/multi_ugv/multi_all_success_rate.png
outputs/figures/multi_ugv/multi_all_average_reward.png
outputs/figures/multi_ugv/multi_all_average_steps.png
outputs/eval/metrics_csv/multi_summary_all.csv
```

## Generalization Repair Experiments

The stronger project line now uses random hard maps and explicitly reports the
training failures instead of hiding them.  The key correction is that the new
`CoverageAStarMultiUGVAgent` does not read hidden survivor coordinates for UAV
search.  The UAV follows coverage waypoints, and UGVs only plan to survivors
after they are discovered.

New files:

```text
configs/env_multi_hard_random_explore.yaml
configs/env_multi_hard_random_guided.yaml
src/fire_rescue_rl/agents/factorized_q_policy.py
scripts/train_factorized_multi_dqn.py
scripts/plot_generalization_results.py
```

Holdout evaluation on 50 random hard maps, seed 1000:

```text
Joint DQN-BC:              success 0.00, delivered 0.14/3
Factorized DQN-DAgger:     success 0.00, delivered 0.22/3
Guided factorized DQN:     success 0.00, delivered 0.48/3, discovered 2.78/3
DQN UGV + A* UAV:          success 0.04, delivered 0.88/3, discovered 3.00/3
Learned UAV + A* UGV:      success 0.84, delivered 2.76/3
Coverage A* upper bound:   success 1.00, delivered 3.00/3
Oracle A* upper bound:     success 1.00, delivered 3.00/3
```

Important interpretation: the all-neural three-agent controller is still not
strong enough on random hard maps.  The reverse hybrid, A* UAV plus DQN UGV,
was also tested after dedicated UGV-only training, but it reached only 4%
success because the learned ground controller still struggles with long-horizon
pickup and return.  The successful repair is the other hierarchical hybrid:
a learned guided UAV search policy plus A* ground rescue planning.  This is the
honest result to present for a high-quality report.

Generated assets:

```text
outputs/models/factorized_multi/factorized_multi_hard_random_guided_seed0_coverage.pt
outputs/models/factorized_multi/factorized_multi_hard_random_guided_seed0_coverage_ugv.pt
outputs/eval/metrics_csv/multi_generalization_diagnosis_summary.csv
outputs/figures/multi_ugv/generalization/generalization_success_rate.png
outputs/figures/multi_ugv/generalization/generalization_delivered_count.png
outputs/figures/multi_ugv/generalization/generalization_risk_exposure.png
outputs/figures/multi_ugv/generalization/generalization_invalid_ugv.png
outputs/figures/multi_ugv/generalization/guided_factorized_training_metrics.png
outputs/videos/learned_uav_astar_ugv_hard_random_guided_seed1000.gif
outputs/videos/learned_uav_astar_ugv_hard_random_guided_seed1000.mp4
```

## PPO Experiment Line

PPO was implemented as a separate experiment line with three wrappers:

```text
PPO-UAV + A*UGV
A*UAV + PPO-UGV
Full PPO over MultiDiscrete([5, 5, 5])
```

New files:

```text
src/fire_rescue_rl/envs/ppo_wrappers.py
configs/ppo_multi.yaml
configs/ppo_multi_conservative.yaml
scripts/train_multi_ppo.py
scripts/eval_multi_ppo.py
scripts/render_multi_ppo.py
scripts/plot_ppo_results.py
```

Holdout results on 50 random hard maps, seed 1000:

```text
PPO-UAV BC:     success 0.48, delivered 1.94/3
PPO-UAV Best:   success 0.38, delivered 1.84/3
PPO-UGV BC:     success 0.00, delivered 0.32/3
PPO-UGV Best:   success 0.00, delivered 0.26/3
Full PPO BC:    success 0.00, delivered 0.06/3
Full PPO Best:  success 0.00, delivered 0.12/3
```

Interpretation: PPO does not solve the full random hard rescue problem under
the current observation and low-level action design.  PPO-UAV is the strongest
PPO branch, but PPO fine-tuning degraded the BC-initialized policy.  The result
should be reported honestly as a negative/diagnostic PPO experiment, not as the
final winning method.

PPO assets:

```text
outputs/models/ppo_multi/ppo_multi_hard_random_guided_uav_seed0_bc.zip
outputs/models/ppo_multi/ppo_multi_hard_random_guided_uav_seed0_best.zip
outputs/models/ppo_multi/ppo_multi_hard_random_guided_ugv_seed0_best.zip
outputs/models/ppo_multi/ppo_multi_hard_random_guided_full_seed0_best.zip
outputs/eval/metrics_csv/ppo_generalization_summary.csv
outputs/figures/multi_ugv/ppo/ppo_success_rate.png
outputs/figures/multi_ugv/ppo/ppo_delivered_count.png
outputs/figures/multi_ugv/ppo/ppo_training_reward_curves.png
outputs/videos/ppo_uav_bc_success_seed1000.gif
outputs/videos/ppo_uav_bc_success_seed1000.mp4
outputs/videos/ppo_uav_bc_failure_seed1001.gif
```

## MaskablePPO Repair Line

The PPO line was repaired with `sb3-contrib` MaskablePPO.  The important change
is not cosmetic: invalid adjacent moves are masked before action sampling, and
the behavior-cloning stage is also trained with the same action masks.  The
best final model remains hierarchical: MaskablePPO controls UAV search, while
Coverage A* controls the two UGV rescue vehicles.

New/updated files:

```text
scripts/train_multi_maskable_ppo.py
scripts/eval_multi_maskable_ppo.py
scripts/render_multi_maskable_ppo.py
scripts/check_maskable_env.py
scripts/plot_maskable_ppo_results.py
src/fire_rescue_rl/envs/ppo_wrappers.py
```

Final hard-random holdout results over 50 maps, seed 1000:

```text
PPO-UAV BC:                     success 0.48, delivered 1.94/3, risk 19.08
MaskablePPO-UAV BC:             success 0.78, delivered 2.68/3, risk 12.79
Masked BC seed3:                success 0.84, delivered 2.78/3, risk 5.92
Masked BC seed4:                success 0.92, delivered 2.86/3, risk 11.00
Masked BC + conservative PPO:   success 0.92, delivered 2.92/3, risk 5.35
DQN-UAV + A* UGV:               success 0.84, delivered 2.76/3, risk 10.09
Coverage A*:                    success 1.00, delivered 3.00/3, risk 1.56
Oracle A*:                      success 1.00, delivered 3.00/3, risk 0.18
```

Interpretation: action masking and masked BC substantially improved the PPO
branch.  Default PPO fine-tuning still degraded the 92% BC model to 70%, but
true conservative fine-tuning kept 92% success and improved delivery count,
risk, reward, and steps.  PPO-UGV and full low-level MaskablePPO remain weak,
so the honest final claim is: the best learned component is UAV search; UGV
transport is still better handled by explicit safe path planning.

Key assets:

```text
outputs/models/maskable_ppo_multi/maskppo_multi_hard_random_guided_uav_seed4_true_conservative_ft.zip
outputs/eval/metrics_csv/maskable_ppo_generalization_summary.csv
outputs/figures/multi_ugv/maskable_ppo/maskppo_success_rate.png
outputs/figures/multi_ugv/maskable_ppo/maskppo_delivered_count.png
outputs/figures/multi_ugv/maskable_ppo/maskppo_risk_exposure.png
outputs/figures/multi_ugv/maskable_ppo/maskppo_uav_optimization_path.png
outputs/figures/multi_ugv/maskable_ppo/maskppo_final_policy_radar.png
outputs/videos/maskppo_uav_trueft_success_seed1000.gif
outputs/videos/maskppo_uav_trueft_success_seed1000.mp4
outputs/videos/maskppo_uav_trueft_failure_seed1001.gif
```
