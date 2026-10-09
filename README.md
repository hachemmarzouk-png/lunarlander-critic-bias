# Critic bias in continuous LunarLander-v3: DDPG vs TD3, with/without LayerNorm

**Research question:** how do TD3 (clipped double-Q, target smoothing, delayed actor) and Layer Normalization affect the bias of the learned critic, measured against Monte-Carlo returns of the current policy, and the performance of the deterministic policy?

DDPG and TD3 are implemented directly in PyTorch on `gymnasium.make("LunarLander-v3", continuous=True)`. At regular checkpoints, the frozen deterministic actor is rolled out and the critic's predictions `Q1(s_t, a_t)` are compared with the discounted return `G_t` actually obtained until the end of the episode. A 2 × 2 factorial design (algorithm × LayerNorm) replicated over seeds assesses the effects and interaction; two DDPG ablations additionally isolate LN placement in actor or critic. A four-page PDF report (French) is regenerated from the results.

## Installation (Python 3.10+)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

If the Box2D install fails, install SWIG first (`brew install swig` on macOS, `sudo apt install swig` on Debian/Ubuntu, `pip install swig` or conda on Windows), then reinstall `gymnasium[box2d]`.

## Running the experiment

```bash
python train_all.py
```

Defaults: 4 configurations × 5 seeds × 300k environment steps = **20 runs** (add `--ablation` for 10 extra DDPG runs), evaluated every 25k steps on 20 episodes. Runs execute **in parallel** (`--jobs`, default = CPU cores − 1, each run uses one thread). Each run logs to `results/<run>/log.txt` and writes a `DONE` marker when finished: **if the script is interrupted, relaunch the same command** and finished runs are skipped (`--force` reruns everything). At the end it aggregates the results and regenerates `rapport_4_pages.pdf` (`--author "Your Name"`).

Rough cost: ~10 ms per TD3 gradient step on one CPU core, i.e. ~40–50 min per run, so ~12–15 h of CPU in total; divide by the number of parallel jobs.

Pipeline check only (not a scientific result): `python train_all.py --quick --seeds 0 1`.

Single run: `python -m rlbias.train --algorithm td3 --layer-norm yes --seed 2`.
Re-aggregate / rebuild the report by hand:

```bash
python -m rlbias.aggregate --results results --out figures
python make_report.py --results results --figures figures --output rapport_4_pages.pdf --author "Your Name"
```

## Methods

- **Networks.** Actor: MLP (256, 256), ReLU, tanh output. Critic: MLP (256, 256) on concatenated state and action. **LayerNorm** ablation: `nn.LayerNorm` after each hidden linear layer, before ReLU, in actor and critics (and their targets); output layers unnormalized.
- **DDPG.** One critic, target `r + γ(1 − terminated) Q'(s', π'(s'))`, actor updated at every step. This is the re-tuned DDPG of Fujimoto et al. ("OurDDPG"): same architecture, learning rates and Gaussian exploration as TD3, no Ornstein-Uhlenbeck noise or L2 penalty, so that only TD3's mechanisms differ.
- **TD3.** Two critics; target uses the minimum of the two target critics at a smoothed target action (σ = 0.2, clip 0.5); actor and targets updated every 2 critic updates; actor maximizes Q1.
- **Hyperparameters.** γ = 0.99, τ = 0.005, Adam lr 1e-3 (actor and critic), batch 256, uniform replay 300k, 10k random warm-up steps, Gaussian exploration σ = 0.1.
- **Termination vs truncation.** Training masks the bootstrap only on true termination (`terminated`), not on the 1000-step time limit (`truncated`): the critic estimates an infinite-horizon value.

### Monte-Carlo bias measurement

- Every checkpoint: deterministic actor (no noise), 20 evaluation episodes with the same initial seeds for every run and checkpoint.
- **Extended horizon, no exclusion.** A policy that hovers is cut at 1000 steps; a MC return stopped there would miss the tail, and discarding those episodes would remove exactly the policies most prone to over-estimation (selection bias). The evaluation environment therefore runs **1000 + H steps (H = 500, `--eval-tail`)**: performance is the sum of the first 1000 rewards (the official task); `G_t` is computed on the extended rollout for every t < 1000. The residual truncation error is at most γ^H · max|r| / (1 − γ) ≈ 0.7 % of the value scale. `timeout_episodes` counts episodes still flying at step 1000.
- **Metrics** (per checkpoint): `q1_bias` = mean over episodes of mean_t (Q1 − G_t) (signed); `q1_mae`, `q1_rmse` likewise; `q1_rel_bias` = `q1_bias` / mean|G_t| (scale-free; dividing by |E[G]| as in REDQ is unstable here because E[G] crosses 0 between crash and landing). Episodes are equal-weight clusters, not independent states. Same metrics for `q2` and `qmin` (TD3 only); the primary comparison uses Q1 in both algorithms.
- `G_t` is a noisy unbiased sample of Q^π(s_t, a_t) only for complete on-policy rollouts; for still-active rollouts stopped at 1500 steps, the omitted discounted tail introduces a residual approximation error. The signed average is interpretable as an occupancy-weighted bias estimate, but cancellation can hide large positive and negative errors. Policies visit different states, so cross-algorithm differences are not matched state-action estimates.

### Statistics

- Unit = training seed. Final value of a run = mean of its **last 3 checkpoints**.
- Per configuration: mean over seeds with a **95 % percentile bootstrap CI** over seeds (`figures/final_summary.csv`).
- 2 × 2 contrasts (`figures/effects.csv`): main effect of TD3, main effect of LN, interaction, and simple effects, with bootstrap CIs (seeds resampled within each cell). With 5 seeds per cell, intervals are approximate.
- Exploratory curves (`figures/*.png`): mean over seeds with bootstrap percentile bands; the **article figures** (`paper/figure_evolution.pdf`) use standard deviations across seeds instead. Step 0 (random critic) is omitted from bias plots.

## Paper

`paper/main.pdf` is the four-page paper (French), built from all 30 runs:

```bash
python paper/make_figures.py                     # figures + LaTeX tables from the logs
cd paper && latexmk -xelatex main.tex              # needs XeLaTeX (polyglossia, unicode-math)
```

Headline results (final values, mean of 5 seeds): DDPG over-estimates (bias +133), TD3 slightly under-estimates (−20), LayerNorm is associated with strong value over-estimation under DDPG (+1499) but not under TD3 (−12). Simple effects are tested with exact permutation tests (`perm_p` in `figures/effects.csv`), because the bootstrap is anti-conservative with 5 seeds.

## Outputs

- `results/{algorithm}_ln-{yes|no|actor|critic}_seed-{n}/`: `evaluations.csv` (one row per checkpoint), `train_episodes.csv`, `config.json` (exact settings), `log.txt`, `DONE`.
- `figures/`: `return.png`, `bias.png`, `rel_bias.png`, `mae.png`, `final_by_seed.csv`, `final_summary.csv`, `effects.csv`, `ablation_by_seed.csv`, `ablation_contrasts.csv` (nonpaired permutation comparisons).
- `rapport_4_pages.pdf`: four-page report regenerated from the 30 provided runs; the authored discussion remains in `paper/main.tex`.

## Limitations

The final paper distinguishes (1) noisy Monte-Carlo references and a bounded residual tail error (under bounded rewards), (2) limited statistical precision with five independent training seeds per condition, and (3) the lack of a causal ablation isolating the three TD3 mechanisms; the actor-only/critic-only LayerNorm ablation is available for DDPG, but not TD3. Measurements are on one environment at a fixed training budget and hyperparameter setting.

## References

1. Fujimoto, van Hoof, Meger (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML.
2. Lillicrap et al. (2016). *Continuous control with deep reinforcement learning*. ICLR.
3. Ba, Kiros, Hinton (2016). *Layer Normalization*. arXiv:1607.06450.
4. Farama Foundation. *Gymnasium — Lunar Lander*.
5. Chen, Wang, Zhou, Ross (2021). *Randomized Ensembled Double Q-Learning (REDQ)*. ICLR.

## Publish on GitHub

```bash
git init -b main
git add .
git commit -m "LunarLander DDPG/TD3 critic bias study"
gh repo create lunarlander-critic-bias --private --source=. --remote=origin --push
```

Git tracks the reproducibility-critical `results/*/evaluations.csv` and summary CSVs automatically, while ignoring machine-local `config.json` paths and full logs. These additional raw files remain available in the distributed ZIP. GitHub Actions runs the tests on every push.


## Rapport scientifique final (quatre pages)

Le document de référence est `rapport_4_pages.pdf`, compilé depuis
`paper/main.tex` : il est fondé **exclusivement sur les 30 journaux d'entraînement
réels fournis dans `results/`**. Les figures révisées sont vectorielles et les
contrastes comparent des groupes indépendants : la coïncidence des numéros de
graine ne constitue pas un appariement. Le tableau 2 donne les différences de
moyennes et les valeurs $p$ de tests exacts par permutation (5 contre 5). Monte-Carlo est une référence empirique,
non une valeur exacte par état-action.

Régénération sans relancer les entraînements :

```bash
python paper/build_paper.py
```

Cette commande nécessite une installation XeLaTeX/latexmk. Elle regénère les
figures et tableaux via `paper/make_figures.py`, compile `paper/main.tex`, puis
met à jour `rapport_4_pages.pdf`. En l'absence de TeX, `train_all.py` peut
encore produire un PDF simplifié avec `make_report.py` ; ce document de secours
n'est pas la version typographiquement soignée.


## DDPG LayerNorm placement ablation (10 additional observed runs)

With the main 20 runs preserved, add 5 seeds each of `ddpg_ln-actor` and
`ddpg_ln-critic` to obtain the **30-run analysis**:

```bash
python train_all.py --ablation --seeds 0 1 2 3 4 --steps 300000
python paper/build_paper.py
```

Existing completed runs are skipped; `--ablation` changes no existing architecture.
The full paper requires 30 result directories in `results/` and automatically
regenerates `paper/table_ablation_revised.tex` from the archived CSV values.
`figures/ablation_by_seed.csv` and `figures/ablation_contrasts.csv` document the
five per-seed values and unpaired exact permutation tests.

In the new ablation, the final DDPG mean critic biases are +133 (no LN), +352
(actor only), +837 (critic only), and +1,499 (both). These data show a stronger
association of critic-only LN with bias than actor-only LN, but do not identify
the dynamical reason nor establish a causal link from bias alone to control quality.
