
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path
import numpy as np
import torch
from .agents import Agent, AgentConfig
from .replay import ReplayBuffer
from .evaluate import evaluate


TASK_HORIZON = 1000  # official LunarLander-v3 time limit


def make_env(max_episode_steps: int | None = None):
    """Training env: official limit. Evaluation env: longer limit (see evaluate.py)."""
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("Gymnasium missing. Install requirements.txt (Box2D extra).") from exc
    if max_episode_steps is None:
        return gym.make("LunarLander-v3", continuous=True)
    return gym.make("LunarLander-v3", continuous=True, max_episode_steps=max_episode_steps)


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algorithm", choices=("ddpg", "td3"), required=True)
    p.add_argument("--layer-norm", choices=("yes", "no", "actor", "critic"), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--exploration-noise", type=float, default=0.1)
    p.add_argument("--eval-tail", type=int, default=500,
                   help="extra steps simulated after the 1000-step limit to complete MC returns")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="results")
    p.add_argument("--save-models", action="store_true")
    return p.parse_args(argv)


def run(args):
    if args.steps <= 0 or args.eval_every <= 0 or args.eval_episodes <= 0:
        raise ValueError("steps, eval-every, eval-episodes must be positive")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + 77)
    train_env = make_env()
    eval_env = make_env(TASK_HORIZON + args.eval_tail)
    train_env.action_space.seed(args.seed)
    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    max_action = float(train_env.action_space.high[0])
    assert train_env.action_space.shape == (2,), "Continuous LunarLander requires two actions"
    cfg = AgentConfig(algorithm=args.algorithm, layer_norm=args.layer_norm != "no",
                      ln_where=args.layer_norm if args.layer_norm in ("actor", "critic") else "both",
                      device=args.device)
    agent = Agent(obs_dim, act_dim, max_action, cfg)
    replay = ReplayBuffer(obs_dim, act_dim, args.buffer_size, args.seed + 5)
    label = f"{args.algorithm}_ln-{args.layer_norm}_seed-{args.seed}"
    run_dir = Path(args.output) / label
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = dict(vars(args), agent_config=vars(cfg), environment="LunarLander-v3",
                    continuous=True, mc_policy="deterministic_actor",
                    mc_horizon=TASK_HORIZON, mc_tail=args.eval_tail,
                    mc_truncation="extended_rollout_no_exclusion", bootstrap_on_truncation=True)
    (run_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    state, _ = train_env.reset(seed=args.seed)
    episode = 0
    episode_return = 0.0
    episode_logs = []
    evaluations = []

    def checkpoint(step):
        row = evaluate(eval_env, agent, args.eval_episodes, cfg.gamma, seed_base=123_000,
                       horizon=TASK_HORIZON, tail=args.eval_tail)
        row.update(algorithm=args.algorithm, layer_norm=args.layer_norm, seed=args.seed, step=step)
        evaluations.append(row)
        keys = list(evaluations[0])
        with (run_dir / "evaluations.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(evaluations)
        print(f"[{label}] step={step} reward={row['mean_return']:.1f} "
              f"q1_bias={row['q1_bias']:.1f} rel={row['q1_rel_bias']:+.2f} "
              f"timeouts={int(row['timeout_episodes'])}/{args.eval_episodes}", flush=True)
        if args.save_models:
            agent.save(str(run_dir / f"checkpoint_{step}.pt"))

    try:
        checkpoint(0)  # random network snapshot, before interacting with environment
        for step in range(1, args.steps + 1):
            if step <= args.warmup:
                action = train_env.action_space.sample()
            else:
                action = np.clip(agent.act(state) + rng.normal(0, args.exploration_noise, size=act_dim),
                                 -max_action, max_action).astype(np.float32)
            next_state, reward, terminated, truncated, _ = train_env.step(action)
            replay.add(state, action, reward, next_state, terminated)
            state = next_state
            episode_return += float(reward)
            if step > args.warmup and replay.size >= args.batch_size:
                agent.update(replay.sample(args.batch_size, args.device))
            if terminated or truncated:
                episode_logs.append({"episode": episode, "step": step, "return": episode_return})
                episode += 1
                episode_return = 0.
                state, _ = train_env.reset()  # seeded RNG progresses naturally
            if step % args.eval_every == 0 or step == args.steps:
                checkpoint(step)
        with (run_dir / "train_episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "step", "return"])
            writer.writeheader()
            writer.writerows(episode_logs)
        (run_dir / "DONE").write_text(str(args.steps), encoding="utf8")
        return evaluations
    finally:
        train_env.close()
        eval_env.close()


def main():
    run(arguments())


if __name__ == "__main__":
    main()
