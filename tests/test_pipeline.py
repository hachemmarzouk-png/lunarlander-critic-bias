"""Integration smoke test using a deterministic fake env; NOT scientific LunarLander data."""
import argparse
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from rlbias.train import run
from rlbias.aggregate import load_results, plot_all
from make_report import build


class MockActionSpace:
    shape = (2,)
    high = np.array([1., 1.], dtype=np.float32)
    def seed(self, seed):
        self.rng = np.random.default_rng(seed)
    def sample(self):
        if not hasattr(self, 'rng'):
            self.seed(42)
        return self.rng.uniform(-1, 1, (2,)).astype(np.float32)


class FakeLander:
    observation_space = type('Space', (), {'shape': (3,)})()
    def __init__(self):
        self.action_space = MockActionSpace()
        self.t = 0
    def reset(self, seed=None):
        self.t = 0
        return np.zeros(3, dtype=np.float32), {}
    def step(self, action):
        self.t += 1
        s = np.array([self.t / 4, float(action[0]), float(action[1])], dtype=np.float32)
        reward = 1. - float(np.square(action).sum())
        return s, reward, self.t == 4, False, {}
    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def test_actor_only_and_critic_only_ablation_runs(self):
        import json
        with tempfile.TemporaryDirectory() as root:
            for ln_where in ("actor", "critic"):
                args = argparse.Namespace(algorithm="ddpg", layer_norm=ln_where, seed=0,
                    steps=12, eval_every=12, eval_episodes=2, warmup=8,
                    batch_size=4, buffer_size=150, exploration_noise=.1,
                    device="cpu", output=str(Path(root)/"results"), save_models=False,
                    eval_tail=3)
                with patch('rlbias.train.make_env', side_effect=lambda *a, **k: FakeLander()):
                    outputs = run(args)
                self.assertEqual(len(outputs), 2)
                run_dir = Path(root)/"results"/f"ddpg_ln-{ln_where}_seed-0"
                meta = json.loads((run_dir/'config.json').read_text())
                self.assertEqual(meta['agent_config']['ln_where'], ln_where)
                self.assertTrue(meta['agent_config']['layer_norm'])
                self.assertEqual(outputs[-1]['layer_norm'], ln_where)

    def test_training_aggregation_and_files(self):
        with tempfile.TemporaryDirectory() as root:
            for alg in ('ddpg', 'td3'):
                args = argparse.Namespace(algorithm=alg, layer_norm='yes', seed=0,
                    steps=24, eval_every=12, eval_episodes=2, warmup=8,
                    batch_size=4, buffer_size=150, exploration_noise=.1,
                    device='cpu', output=str(Path(root)/'results'), save_models=False,
                    eval_tail=3)
                with patch('rlbias.train.make_env', side_effect=lambda *a, **k: FakeLander()):
                    outputs = run(args)
                self.assertEqual(len(outputs), 3)
                self.assertTrue(all(r['complete_episodes'] == 2 for r in outputs))
            df = load_results(Path(root)/'results')
            self.assertEqual(len(df), 6)
            self.assertEqual(len(plot_all(df, Path(root)/'figures')), 2)
            for name in ('bias.png', 'return.png', 'mae.png', 'final_by_seed.csv'):
                self.assertTrue((Path(root)/'figures'/name).exists())
            pdf = Path(root)/'mock_only.pdf'
            build(pdf, Path(root)/'results', Path(root)/'figures')
            self.assertTrue(pdf.exists())


if __name__ == '__main__':
    unittest.main()
