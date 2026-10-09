"""On-policy deterministic rollouts used as Monte-Carlo references for a frozen actor.

Horizon handling
----------------
LunarLander-v3 stops episodes after 1000 steps (time-limit truncation). The critic is
trained with bootstrapping through truncation, so it estimates the *infinite-horizon*
discounted value Q^pi(s, a), which has no notion of "steps remaining".

A plain Monte-Carlo sum stopped at step 1000 would therefore miss the tail of the
return for late states, and simply discarding truncated episodes would remove exactly
the hovering policies where over-estimation is most likely (selection bias).

Instead, the evaluation environment is allowed to run ``horizon + tail`` steps:
  * performance  = undiscounted sum of the first ``horizon`` rewards (the official task);
  * critic bias  = Q(s_t, a_t) - G_t for every t < horizon, where G_t is the discounted
    return computed on the *extended* rollout. If the rollout was cut, every kept state
    still has at least ``tail`` future steps, so the missing part is at most
    gamma**tail * max|r| / (1 - gamma)  (about 0.7% of the scale for tail=500, gamma=0.99).
No episode is excluded. States whose observed future is shorter than ``tail`` (only
possible if the environment is cut earlier than horizon + tail) are dropped and counted.

G_t is a single noisy sample of Q^pi(s_t, a_t) (engine dispersion in LunarLander is
random), so all metrics are averaged over episodes and seeds.
"""
from __future__ import annotations
import numpy as np


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    out = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = float(rewards[i]) + gamma * running
        out[i] = running
    return out


CRITICS = ("q1", "q2", "qmin")


def evaluate(env, agent, episodes: int, gamma: float, seed_base: int,
             horizon: int = 1000, tail: int = 500) -> dict[str, float]:
    """Roll out the deterministic actor and compare its critic(s) to MC returns.

    ``env`` should allow at least ``horizon + tail`` steps (see make_env in train.py).
    """
    returns: list[float] = []
    lengths: list[int] = []
    timeouts = 0
    per_episode: dict[str, dict[str, list[float]]] = {
        n: {"bias": [], "mae": [], "rmse": []} for n in CRITICS}
    all_errors: dict[str, list[np.ndarray]] = {n: [] for n in CRITICS}
    abs_mc: list[float] = []
    n_kept = n_dropped = 0
    complete = 0

    for ep in range(episodes):
        state, _ = env.reset(seed=seed_base + ep)
        states, actions, rewards = [], [], []
        terminated = truncated = False
        while not (terminated or truncated) and len(rewards) < horizon + tail:
            action = agent.act(state)
            states.append(np.array(state, copy=True))
            actions.append(np.array(action, copy=True))
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))

        length = len(rewards)
        returns.append(float(sum(rewards[:horizon])))      # official task performance
        lengths.append(min(length, horizon))
        if not (terminated and length <= horizon):
            timeouts += 1                                  # still flying at the time limit

        reference = discounted_returns(rewards, gamma)
        t = np.arange(length)
        keep = t < horizon
        if not terminated:                                 # need a long enough observed future
            keep &= (length - t) >= tail
        n_kept += int(keep.sum())
        n_dropped += int((t < horizon).sum() - keep.sum())
        if not keep.any():
            continue
        complete += 1
        g = reference[keep]
        abs_mc.append(float(np.mean(np.abs(g))))
        q_values = agent.predict_q(np.asarray(states)[keep], np.asarray(actions)[keep])
        for name, q in q_values.items():
            delta = np.asarray(q, dtype=np.float64) - g
            # Each episode is one cluster: average within episode first.
            per_episode[name]["bias"].append(float(np.mean(delta)))
            per_episode[name]["mae"].append(float(np.mean(np.abs(delta))))
            per_episode[name]["rmse"].append(float(np.sqrt(np.mean(delta ** 2))))
            all_errors[name].append(delta)

    scale = float(np.mean(abs_mc)) if abs_mc else float("nan")
    row: dict[str, float] = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns, ddof=1)) if episodes > 1 else 0.,
        "mean_length": float(np.mean(lengths)),
        "total_episodes": float(episodes),
        "timeout_episodes": float(timeouts),
        "complete_episodes": float(complete),
        "evaluated_states": float(n_kept),
        "dropped_states": float(n_dropped),
        "mc_abs_scale": scale,
    }
    for name in CRITICS:
        for metric in ("bias", "mae", "rmse"):
            values = per_episode[name][metric]
            row[f"{name}_{metric}"] = float(np.mean(values)) if values else float("nan")
            row[f"{name}_{metric}_sd"] = (float(np.std(values, ddof=1))
                                          if len(values) > 1 else float("nan"))
        # Relative (scale-free) bias: E[Q - G] / E|G|. Unlike dividing by |E[G]|
        # (REDQ, Chen et al. 2021), the denominator cannot approach 0 when
        # crash (-) and landing (+) returns cancel out mid-training.
        if per_episode[name]["bias"] and scale > 0:
            row[f"{name}_rel_bias"] = row[f"{name}_bias"] / scale
            row[f"{name}_rel_sd"] = float(np.std(np.concatenate(all_errors[name]))) / scale
        else:
            row[f"{name}_rel_bias"] = row[f"{name}_rel_sd"] = float("nan")
    return row
