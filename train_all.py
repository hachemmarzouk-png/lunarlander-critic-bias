"""Train DDPG/TD3 across independent seeds (optional LN placement ablation)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run_one(cmd: list[str], run_dir: Path) -> tuple[str, int, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with (run_dir / "log.txt").open("w", encoding="utf8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
    return run_dir.name, code, time.monotonic() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--output", default="runs", help="training logs; keep the published results/ unchanged")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--ablation", action="store_true", help="also test actor-only/critic-only LN in DDPG")
    parser.add_argument("--force", action="store_true", help="rerun completed experiments")
    parser.add_argument("--quick", action="store_true", help="short smoke run, not a scientific result")
    args = parser.parse_args()
    if args.quick:
        args.steps, args.eval_every, args.eval_episodes = 1500, 500, 2
        if args.output == "runs":
            args.output = "runs_quick"
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output

    configurations = [(alg, ln) for alg in ("ddpg", "td3") for ln in ("no", "yes")]
    if args.ablation:
        configurations += [("ddpg", "actor"), ("ddpg", "critic")]

    jobs = []
    for seed in args.seeds:
        for algorithm, ln in configurations:
            run_dir = output / f"{algorithm}_ln-{ln}_seed-{seed}"
            done = run_dir / "DONE"
            if not args.force and done.exists() and done.read_text().strip() == str(args.steps):
                print("Already completed:", run_dir.name)
                continue
            cmd = [sys.executable, "-m", "rlbias.train", "--algorithm", algorithm,
                   "--layer-norm", ln, "--seed", str(seed), "--steps", str(args.steps),
                   "--eval-every", str(args.eval_every), "--eval-episodes", str(args.eval_episodes),
                   "--output", str(output), "--device", args.device]
            if args.quick:
                cmd += ["--warmup", "500", "--batch-size", "32", "--eval-tail", "100"]
            jobs.append((cmd, run_dir))

    print(f"{len(jobs)} runs to execute using {args.jobs} workers.", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, cmd, path) for cmd, path in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            name, code, seconds = future.result()
            print(f"[{index}/{len(jobs)}] {name}: {'OK' if code == 0 else 'FAILED'} ({seconds/60:.1f} min)", flush=True)
            if code:
                failures.append(name)
    if failures:
        raise SystemExit("Failed runs (see their log.txt): " + ", ".join(failures))


if __name__ == "__main__":
    main()
