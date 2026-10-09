"""Tests without Box2D: the algorithm must be independently verifiable."""
import unittest
import numpy as np
import torch
from torch import nn
from rlbias.agents import Agent, AgentConfig
from rlbias.evaluate import discounted_returns, evaluate
from rlbias.replay import ReplayBuffer


class ConstantCritic(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)
    def forward(self, state, action):
        return torch.ones((state.shape[0], 1)) * self.value


class MockAgent:
    def act(self, s):
        return np.zeros(2, dtype=np.float32)
    def predict_q(self, ss, aa):
        return {"q1": np.full(len(ss), 3.0)}


class MockEnv:
    def __init__(self, truncated=False):
        self.truncated = truncated
        self.i = 0
    def reset(self, seed=None):
        self.i = 0
        return np.zeros(3, dtype=np.float32), {}
    def step(self, action):
        self.i += 1
        return np.ones(3, dtype=np.float32), 1.0, self.i == 3 and not self.truncated, self.i == 3 and self.truncated, {}


class LongEnv:
    """Never terminates by itself; reward 1 per step."""
    def __init__(self):
        self.steps_taken = 0
    def reset(self, seed=None):
        self.steps_taken = 0
        return np.zeros(3, dtype=np.float32), {}
    def step(self, action):
        self.steps_taken += 1
        return np.ones(3, dtype=np.float32), 1.0, False, False, {}


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.s = torch.randn(8, 3)
        self.a = torch.randn(8, 2).clamp(-1, 1)
        self.r = torch.ones(8, 1)
        self.term = torch.zeros(8, 1)
        self.term[0] = 1.

    def test_monte_carlo_recursion(self):
        np.testing.assert_allclose(discounted_returns([1, 2, 3], .5), [2.75, 3.5, 3])

    def test_ddpg_target_and_terminal(self):
        agent = Agent(3, 2, 1., AgentConfig(algorithm="ddpg", gamma=.9))
        agent.critic1_target = ConstantCritic(10)
        y = agent.bellman_target(self.r, self.s, self.term)
        self.assertAlmostEqual(float(y[0]), 1.)
        self.assertAlmostEqual(float(y[1]), 10.)

    def test_td3_minimum_target(self):
        agent = Agent(3, 2, 1., AgentConfig(algorithm="td3", gamma=.9))
        agent.critic1_target = ConstantCritic(10)
        agent.critic2_target = ConstantCritic(4)
        y = agent.bellman_target(self.r, self.s, self.term)
        self.assertAlmostEqual(float(y[1]), 4.6, places=5)

    def test_layer_norm_both_networks(self):
        agent = Agent(3, 2, 1., AgentConfig(layer_norm=True))
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.actor.modules()), 2)
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.critic1.modules()), 2)

    def test_layer_norm_actor_only(self):
        agent=Agent(3,2,1., AgentConfig(algorithm="ddpg", layer_norm=True,ln_where="actor"))
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.actor.modules()), 2)
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.critic1.modules()), 0)

    def test_layer_norm_critic_only(self):
        agent=Agent(3,2,1., AgentConfig(algorithm="ddpg", layer_norm=True,ln_where="critic"))
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.actor.modules()), 0)
        self.assertEqual(sum(isinstance(m, nn.LayerNorm) for m in agent.critic1.modules()), 2)

    def test_ddpg_and_td3_update_schedule(self):
        batch = (self.s, self.a, self.r, self.s, self.term)
        ddpg = Agent(3, 2, 1., AgentConfig(algorithm="ddpg", hidden=(16, 16)))
        td3 = Agent(3, 2, 1., AgentConfig(algorithm="td3", hidden=(16, 16)))
        self.assertEqual(ddpg.update(batch)["actor_updated"], 1.)
        self.assertEqual(td3.update(batch)["actor_updated"], 0.)
        self.assertEqual(td3.update(batch)["actor_updated"], 1.)

    def test_replay_sampling(self):
        b = ReplayBuffer(3, 2, 10, 1)
        b.add(np.zeros(3), np.zeros(2), 4., np.ones(3), False)
        self.assertEqual(b.sample(2)[0].shape, (2, 3))

    def test_mc_evaluation_terminated(self):
        row = evaluate(MockEnv(), MockAgent(), 2, gamma=.9, seed_base=0, horizon=10, tail=5)
        self.assertEqual(row["complete_episodes"], 2.)
        self.assertEqual(row["timeout_episodes"], 0.)
        self.assertEqual(row["dropped_states"], 0.)
        g = [2.71, 1.9, 1.]
        self.assertAlmostEqual(row["q1_bias"], 3 - sum(g) / 3)
        self.assertAlmostEqual(row["mc_abs_scale"], sum(g) / 3)
        self.assertAlmostEqual(row["q1_rel_bias"], (3 - sum(g) / 3) / (sum(g) / 3))

    def test_mc_extended_horizon_keeps_hovering_episodes(self):
        # Never-terminating env: 'hovering' policy. Task horizon 4, tail 6.
        env = LongEnv()
        row = evaluate(env, MockAgent(), 1, gamma=.5, seed_base=0, horizon=4, tail=6)
        self.assertEqual(env.steps_taken, 10)            # rollout extended to horizon + tail
        self.assertAlmostEqual(row["mean_return"], 4.)   # performance: first 4 rewards only
        self.assertEqual(row["timeout_episodes"], 1.)
        self.assertEqual(row["evaluated_states"], 4.)    # all t < horizon kept, none excluded
        self.assertEqual(row["dropped_states"], 0.)
        g = [sum(.5 ** k for k in range(10 - t)) for t in range(4)]
        self.assertAlmostEqual(row["q1_bias"], 3 - np.mean(g))

    def test_mc_short_env_drops_states_without_enough_future(self):
        # Env cut at 3 steps (not terminal) while tail=2 is required: only t=0,1 usable.
        row = evaluate(MockEnv(truncated=True), MockAgent(), 2, .9, 0, horizon=10, tail=2)
        self.assertEqual(row["evaluated_states"], 4.)
        self.assertEqual(row["dropped_states"], 2.)
        self.assertAlmostEqual(row["mean_return"], 3.)
        self.assertAlmostEqual(row["q1_bias"], 3 - (2.71 + 1.9) / 2)

class StatsTests(unittest.TestCase):
    def test_exact_permutation_test(self):
        from rlbias.aggregate import permutation_pvalue
        # Complete separation between two independent groups of five.
        self.assertAlmostEqual(permutation_pvalue([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), 2 / 252)

    def test_archived_experiments(self):
        from pathlib import Path
        from rlbias.aggregate import load_results, final_by_seed, permutation_pvalue
        data = final_by_seed(load_results(Path(__file__).resolve().parents[1] / "results"))
        self.assertEqual(len(data), 30)

        def values(algorithm, ln, metric):
            return data[(data.algorithm == algorithm) & (data.layer_norm == ln)][metric].to_numpy()

        p_td3 = permutation_pvalue(values("td3", "no", "mean_return"),
                                   values("ddpg", "no", "mean_return"))
        p_ln = permutation_pvalue(values("td3", "yes", "mean_return"),
                                  values("td3", "no", "mean_return"))
        p_ablation = permutation_pvalue(values("ddpg", "yes", "q1_bias"),
                                       values("ddpg", "critic", "q1_bias"))
        self.assertAlmostEqual(p_td3, 2 / 252)
        self.assertAlmostEqual(p_ln, 18 / 252)
        self.assertAlmostEqual(p_ablation, 12 / 252)


if __name__ == "__main__":
    unittest.main()
