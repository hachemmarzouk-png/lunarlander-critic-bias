"""Run the full 2x2 design (DDPG/TD3 x with/without LayerNorm) over several seeds.

Default: 4 configurations x 5 seeds x 300k environment steps = 20 runs.
Runs execute in parallel (one CPU thread each) and finished runs are skipped,
so the script can simply be relaunched after an interruption.
Use --quick only to test the pipeline, never as evidence of trained performance.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_one(cmd: list[str], run_dir: Path) -> tuple[str, int, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with (run_dir / "log.txt").open("w", encoding="utf8") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE).returncode
    return run_dir.name, code, time.time() - start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--eval-every", type=int, default=25_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--output", default="results")
    p.add_argument("--figures", default="figures")
    p.add_argument("--report", default="rapport_4_pages.pdf")
    p.add_argument("--author", default="Hachem")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="number of runs in parallel (default: number of CPU cores - 1)")
    p.add_argument("--ablation", action="store_true",
                   help="also train DDPG with LN only in actor or only in critic (10 extra runs)")
    p.add_argument("--force", action="store_true", help="rerun even finished runs")
    p.add_argument("--quick", action="store_true", help="Fast smoke run, not a scientific result")
    a = p.parse_args()
    if a.quick:
        a.steps, a.eval_every, a.eval_episodes = 1500, 500, 2

    output = (HERE / a.output) if not Path(a.output).is_absolute() else Path(a.output)
    jobs = []
    # Seed-major order: if interrupted, each condition is advanced seed by seed.
    for seed in a.seeds:
        grid = [(alg, ln) for alg in ("ddpg", "td3") for ln in ("no", "yes")]
        if a.ablation:
            grid += [("ddpg", "actor"), ("ddpg", "critic")]
        for algorithm, ln in grid:
            run_dir = output / f"{algorithm}_ln-{ln}_seed-{seed}"
            done = run_dir / "DONE"
            if not a.force and done.exists() and done.read_text().strip() == str(a.steps):
                print(f"skip (already finished): {run_dir.name}")
                continue
            cmd = [sys.executable, "-m", "rlbias.train", "--algorithm", algorithm,
                   "--layer-norm", ln, "--seed", str(seed),
                   "--steps", str(a.steps), "--eval-every", str(a.eval_every),
                   "--eval-episodes", str(a.eval_episodes), "--output", str(output),
                   "--device", a.device]
            if a.quick:
                cmd += ["--warmup", "500", "--batch-size", "32", "--eval-tail", "100"]
            jobs.append((cmd, run_dir))
    print(f"{len(jobs)} run(s) to execute, {a.jobs} in parallel. "
          f"Progress: tail -f {a.output}/<run>/log.txt", flush=True)
    failures = []
    start = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(run_one, cmd, run_dir) for cmd, run_dir in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            name, code, dt = fut.result()
            status = "ok" if code == 0 else f"FAILED (code {code}, see log.txt)"
            print(f"[{i}/{len(jobs)}] {name}: {status} in {dt/60:.1f} min "
                  f"(elapsed {(time.time()-start)/3600:.2f} h)", flush=True)
            if code != 0:
                failures.append(name)
    if failures:
        print("Failed runs:", ", ".join(failures), "- relaunch the same command to retry them.")
        sys.exit(1)

    subprocess.run([sys.executable, "-m", "rlbias.aggregate", "--results", str(output),
                    "--out", a.figures], check=True, cwd=HERE)
    # Prefer the revised four-page LaTeX report when TeX Live is available.
    # The ReportLab report is kept as a no-TeX fallback for lightweight machines.
    if shutil.which("latexmk") and shutil.which("xelatex"):
        subprocess.run([sys.executable, "paper/build_paper.py"], check=True, cwd=HERE)
        final_report = HERE / "rapport_4_pages.pdf"
        requested_report = (HERE / a.report) if not Path(a.report).is_absolute() else Path(a.report)
        if final_report.resolve() != requested_report.resolve():
            shutil.copy2(final_report, requested_report)
    else:
        print("XeLaTeX not installed: generating the fallback report instead.", flush=True)
        subprocess.run([sys.executable, "make_report.py", "--results", str(output),
                        "--figures", a.figures, "--output", a.report, "--author", a.author],
                       check=True, cwd=HERE)


if __name__ == "__main__":
    main()
