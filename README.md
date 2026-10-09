# DDPG, TD3 and LayerNorm — critic bias on LunarLander-v3

**Hachem Marzouk · M2 MS2A, Sorbonne Université**

**[Scientific report (PDF, 4 pages)](rapport_4_pages.pdf)**

We compare the Monte-Carlo estimation bias of DDPG and TD3 critics and the returns
of their deterministic policies on continuous `LunarLander-v3`. LayerNorm is
examined both jointly and separately in the DDPG actor and critic.

**Data:** 30 independent trainings (six configurations × five seeds), 300,000
environment steps per run. The complete evaluation checkpoints used in the paper
are included in `results/`; no retraining is needed to verify the results.

## Reproduce the analysis

Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python paper/make_figures.py
```

The last command regenerates the paper's two figures and three tables from
`results/*/evaluations.csv`. To recompile `rapport_4_pages.pdf` as well, install
XeLaTeX and `latexmk`, then run:

```bash
python paper/build_paper.py
```

The ready-to-read PDF is included in the repository; TeX is optional.
On macOS, Box2D installation may require `brew install swig`.

## Code and data

| Location | Role |
| --- | --- |
| `rapport_4_pages.pdf` | Final scientific report |
| `results/*/evaluations.csv` | 30 archived evaluation histories, one per independent run |
| `rlbias/` | DDPG/TD3, replay buffer, evaluation and statistics |
| `train_all.py` | Optional training launcher |
| `paper/` | LaTeX source and figure/table generators |
| `tests/` | Algorithm and training/evaluation pipeline tests |

## Training (optional)

```bash
python train_all.py --ablation --seeds 0 1 2 3 4 --steps 300000
```

This starts 30 **new, computationally expensive** trainings in `runs/`, separate
from the archived `results/`. Use `--jobs 2` to limit parallelism; `--quick`
is for smoke tests only (saved in `runs_quick/`). Hyperparameters and the
statistical methodology are documented in the report and source code.

## References

Fujimoto et al. (2018), *Addressing Function Approximation Error in Actor-Critic Methods*;
Lillicrap et al. (2016), *Continuous Control with Deep Reinforcement Learning*;
Ba et al. (2016), *Layer Normalization*.
