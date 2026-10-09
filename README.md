# Critic estimation bias in LunarLander-v3 — DDPG, TD3 and LayerNorm

**Experimental reinforcement learning project · Hachem Marzouk · M2 MS2A, Sorbonne Université**

**[Read the four-page scientific report (PDF)](rapport_4_pages.pdf)** · [LaTeX source](paper/main.tex)

This study compares the signed estimation bias of the DDPG and TD3 critics against Monte-Carlo returns, alongside the performance of their deterministic policies in continuous-control LunarLander-v3. It also examines Layer Normalization (LN), including whether LN is applied to the actor, the critic, or both in DDPG.

The repository includes **30 independent training runs**: six configurations × five seeds, each trained for **300,000 environment steps**. Evaluation uses 20 episodes every 25,000 steps. The report and the recorded experiment results are provided; reproducing the analysis does **not** require retraining the agents.

## Reproduce the analysis (no retraining)

Requires Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Regenerate the analysis figures and tables directly from the archived `results/*/evaluations.csv` files:

```bash
python paper/make_figures.py
```

To rebuild the **four-page PDF** from the data and LaTeX source:

```bash
python paper/build_paper.py
```

This last command additionally requires **XeLaTeX** and **`latexmk`**. The finished PDF is already available above, so installing TeX is not necessary to read the report or inspect the results. If Box2D installation fails on macOS, install SWIG with `brew install swig`, then retry installing the requirements.

## Experimental design

| Configuration | LayerNorm placement | Seeds |
| --- | --- | ---: |
| DDPG | None | 5 |
| DDPG + LN | Actor and critic | 5 |
| TD3 | None | 5 |
| TD3 + LN | Actor and both critics | 5 |
| DDPG, actor-only LN | Actor | 5 |
| DDPG, critic-only LN | Critic | 5 |

- **Environment:** `gymnasium.make("LunarLander-v3", continuous=True)`.
- **Models:** PyTorch actor and critic MLPs with two 256-unit hidden layers. DDPG uses one critic; TD3 uses two critics, target-policy smoothing and delayed actor updates.
- **Training:** `γ = 0.99`, `τ = 0.005`, batch size 256, 10,000-step random warm-up. Other parameters and implementation details are in the training code.
- **Evaluation:** freeze the deterministic actor at each checkpoint; compare `Q₁(sₜ, aₜ)` with the discounted Monte-Carlo return `Gₜ`. Episode-averaged signed bias and absolute error are reported separately from the undiscounted episode return. Evaluation rollouts may continue beyond the 1,000-step performance horizon to reduce truncated-return bias.
- **Statistics:** each *independent training run*, not each visited state, is one observational unit. Final statistics average the last three evaluation checkpoints. Group comparisons use two-sided exact permutation tests (five runs versus five, 252 possible reallocations). These comparisons are exploratory given the small sample size and multiple tests.

The report discusses the observed bias and performance differences as well as the limits of interpreting them causally.

## Repository contents

| Path | Contents |
| --- | --- |
| [`rapport_4_pages.pdf`](rapport_4_pages.pdf) | Final scientific report (French) |
| [`paper/`](paper/) | LaTeX source and scripts to regenerate report figures/tables |
| [`rlbias/`](rlbias/) | DDPG/TD3 agents, networks, training, evaluation and aggregation |
| [`results/`](results/) | Evaluation and episode CSVs for the 30 training runs |
| [`figures/`](figures/) | Summary tables, contrasts and plotted results |
| [`tests/`](tests/) | Automated tests |
| [`train_all.py`](train_all.py) | Full training pipeline |
| [`requirements.txt`](requirements.txt) | Python dependencies |

## Retrain the agents (optional, computationally expensive)

To run all six configurations with five seeds each:

```bash
python train_all.py --ablation --seeds 0 1 2 3 4 --steps 300000
```

Runs are parallelized (`--jobs` controls concurrency). Completed local runs marked with `DONE` are skipped on reruns. **Important:** the GitHub repository contains the experimental CSVs but not the local `DONE` markers or full training logs; running this command on a fresh clone will start new training runs and can replace the existing results. You do **not** need to run it to inspect or regenerate the paper.

For a fast pipeline smoke test only (not a scientific result), use `python train_all.py --quick --seeds 0 1` in a separate working copy.

## Scope and limitations

The Monte-Carlo comparison depends on the visited state-action distribution and has sampling noise; a small signed bias can also conceal large errors of opposite signs. The experiments use one environment, a fixed training budget and five independent seeds per configuration. The DDPG actor-only/critic-only LN ablation helps locate associations but does not establish the dynamical mechanism behind the bias; TD3's constituent modifications are not individually ablated.

## References

- Fujimoto, van Hoof & Meger (2018), *Addressing Function Approximation Error in Actor-Critic Methods*.
- Lillicrap et al. (2016), *Continuous Control with Deep Reinforcement Learning*.
- Ba, Kiros & Hinton (2016), *Layer Normalization*.
- [Gymnasium — Lunar Lander](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
