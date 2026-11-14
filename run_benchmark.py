"""Full optimizer benchmark on the GPU: GA, then PSO, then DE, then GWO, five seeds each.

    python run_benchmark.py                   # full benchmark -> results/benchmark_<stamp>/
    python run_benchmark.py --validate-only   # CPU vs GPU check only
    python run_benchmark.py --smoke           # tiny budget, checks the whole pipeline

Run it on the `results` branch. Each invocation writes to a new folder and never
touches earlier runs:

    results/benchmark_<YYYYmmdd_HHMMSS>/
        manifest.json          settings, versions, GPU, git commit, timings, checks
        benchmark.log          everything printed during the run
        validation/            CPU (filterpy) vs batched GPU EKF on GA seed 1, generation 1
        baseline/              THETA0 through the same fitness (reference)
        GA/ PSO/ DE/ GWO/      runs.csv (one row per seed), convergence_seed<s>.csv, live logs
        results_table.*, convergence.png, robustness.png, response_4d.*   (plot_response.py)

Order of work:
    0. checks: results branch, CUDA GPU, identical budget for all four optimizers
    1. validation: the first generation GA evaluates for seed 1 (200 candidates) is
       scored by the GPU through the real optimizer path and again by the filterpy
       reference; the run stops if any difference exceeds ekf.gpu.TOL_RMSE
    2. baseline
    3. GA seeds 1-5, saved; PSO seeds 1-5, saved; DE seeds 1-5, saved; GWO seeds 1-5, saved.
       One optimizer at a time, in this process, so the GPU serves one optimizer only.
       After each optimizer, every seed is checked for n_evals == n_objective_evals == max_evals.
    4. table and figures
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import ekf

ORDER = ["GA", "PSO", "DE", "GWO"]
FULL = {"population_size": 200, "max_evals": 10000, "seeds": (1, 2, 3, 4, 5)}
SMOKE = {"population_size": 4, "max_evals": 8, "seeds": (1,)}
LOCK = Path("results") / ".benchmark.lock"


class Tee:
    """Copy everything written to stdout into benchmark.log."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def git(*args):
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def optimizer_modules():
    import DE, DE_config, GA, GA_config, GWO, GWO_config, PSO, PSO_config
    return {"GA": (GA, GA_config), "PSO": (PSO, PSO_config), "DE": (DE, DE_config), "GWO": (GWO, GWO_config)}


def config_values(cfg):
    return {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}


def population_of(cfg):
    return getattr(cfg, "POPULATION_SIZE", getattr(cfg, "SWARM_SIZE", None))


def check_configs(modules):
    """Every optimizer must use the benchmark budget, so the comparison is fair."""
    problems = []
    for name, (_, cfg) in modules.items():
        got = {"population_size": population_of(cfg), "max_evals": cfg.MAX_EVALS, "seeds": tuple(cfg.SEEDS)}
        for key, want in FULL.items():
            if got[key] != want:
                problems.append(f"{name}: {key} = {got[key]}, benchmark needs {want}")
        if getattr(cfg, "BACKEND", None) != "gpu":
            problems.append(f"{name}: BACKEND = {getattr(cfg, 'BACKEND', None)}, benchmark needs 'gpu'")
    if problems:
        raise SystemExit("Config mismatch:\n  " + "\n  ".join(problems))


def validate(data, modules, out_dir, population_size, n_workers):
    """GA seed 1, generation 1: optimizer -> tracker -> GPU, then the filterpy reference."""
    from ekf.gpu import TOL_RMSE, compare_with_cpu
    GA, _ = modules["GA"]
    out_dir.mkdir(parents=True)
    print(f"\n=== Validation: GA seed 1, first generation ({population_size} candidates) ===", flush=True)
    with ekf.make_cost_fn(data, max_evals=population_size, backend="gpu") as tracker:
        GA.run_ga(data, tracker, seed=1, population_size=population_size)
        hist = tracker.convergence()
    thetas = hist[ekf.THETA_NAMES].to_numpy()
    print(f"GPU scored {len(thetas)} candidates through the optimizer; scoring them again with filterpy "
          f"on {n_workers} CPU workers...", flush=True)
    table, summary = compare_with_cpu(data, thetas, n_workers=n_workers)
    table.insert(0, "tracker_fitness", hist["fitness"].to_numpy())
    table["diff_tracker_vs_cpu"] = np.abs(table["tracker_fitness"] - table["cpu_rmse_h"])
    summary["max_diff_tracker_vs_cpu"] = float(table["diff_tracker_vs_cpu"].max())
    summary["passed"] = bool(summary["passed"] and summary["max_diff_tracker_vs_cpu"] <= TOL_RMSE)
    summary.update(optimizer="GA", seed=1, generation=1)
    table.to_csv(out_dir / "validation_table.csv", index=False)
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"max |GPU - CPU| fitness {summary['max_diff_fitness']:.3e} m, RMSE_v {summary['max_diff_rmse_v']:.3e} m, "
          f"RMSE_3D {summary['max_diff_rmse_3d']:.3e} m, mean NIS (rel.) {summary['max_reldiff_mean_nis']:.3e}; "
          f"failed candidates agree: {summary['ok_flags_agree']} -> "
          f"{'PASSED' if summary['passed'] else 'FAILED'} (tolerance {TOL_RMSE:g} m)", flush=True)
    return summary


def run_baseline(data, out_dir):
    print("\n=== Baseline (THETA0, filterpy reference) ===", flush=True)
    rmse_gps = ekf.gps_rmse(data)
    with ekf.make_cost_fn(data, max_evals=1) as tracker:
        rmse_baseline = tracker(ekf.THETA0)
        row, _ = ekf.run_summary(data, tracker, "baseline", 0, rmse_gps=rmse_gps, rmse_baseline=rmse_baseline,
                                 population_size=1)
        ekf.save_run(out_dir, row, tracker)
    print(f"Baseline RMSE_h {rmse_baseline:.4f} m, raw GPS {rmse_gps:.4f} m", flush=True)
    return {"rmse_h": rmse_baseline, "rmse_gps": rmse_gps}


def check_runs(out_dir, name, seeds, max_evals, population_size):
    """Evaluation accounting: each seed must have used exactly max_evals evaluations."""
    runs = pd.read_csv(out_dir / "runs.csv")
    checks = []
    for seed in seeds:
        r = runs[runs["seed"] == seed]
        conv = out_dir / f"convergence_seed{seed}.csv"
        n_curve = len(pd.read_csv(conv)) if conv.exists() else 0
        c = {"optimizer": name, "seed": seed, "present": len(r) == 1, "convergence_rows": n_curve}
        if len(r) == 1:
            r = r.iloc[0]
            c.update(n_evals=int(r["n_evals"]), n_objective_evals=int(r["n_objective_evals"]),
                     n_generations=int(r["n_generations"]), population_size=int(r["population_size"]),
                     rmse_h=float(r["rmse_h"]), search_vs_final_diff=float(r["search_vs_final_diff"]))
        c["ok"] = bool(c["present"] and c.get("n_evals") == max_evals and c.get("n_objective_evals") == max_evals
                       and n_curve == max_evals and c.get("population_size") == population_size)
        checks.append(c)
        status = "ok" if c["ok"] else "MISMATCH"
        print(f"  check {name} seed {seed}: N_eval {c.get('n_evals')}, scored by backend "
              f"{c.get('n_objective_evals')}, curve rows {n_curve}, generations {c.get('n_generations')} -> {status}",
              flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="tiny budget to test the pipeline")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--validate-workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)),
                        help="CPU processes for the filterpy side of the validation")
    parser.add_argument("--allow-any-branch", action="store_true")
    parser.add_argument("--out", help="output folder (default results/benchmark_<stamp>); must not exist")
    args = parser.parse_args()

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "results" and not args.allow_any_branch:
        raise SystemExit(f"run_benchmark.py writes benchmark results to the results branch; current branch is "
                         f"{branch!r}. Check out results or pass --allow-any-branch.")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch; the benchmark needs the GPU backend.")

    modules = optimizer_modules()
    budget = SMOKE if args.smoke else FULL
    if not args.smoke:
        check_configs(modules)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default = f"results/{'smoke' if args.smoke else 'benchmark'}_{stamp}"
    out = Path(args.out or default)
    if out.exists():
        raise SystemExit(f"{out} already exists; the benchmark never overwrites earlier runs.")

    LOCK.parent.mkdir(exist_ok=True)
    try:
        lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{LOCK} exists: another benchmark is running (or one crashed; delete the file "
                         f"if no benchmark is running).")
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)

    try:
        out.mkdir(parents=True)
        sys.stdout = Tee(out / "benchmark.log")
        manifest = {
            "started": datetime.now().isoformat(timespec="seconds"),
            "mode": "smoke" if args.smoke else ("validate-only" if args.validate_only else "full"),
            "order": ORDER, "budget": {**budget, "seeds": list(budget["seeds"])},
            "git": {"branch": branch, "commit": git("rev-parse", "HEAD"),
                    "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "torch": torch.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(0), "platform": platform.platform()},
            "configs": {name: config_values(cfg) for name, (_, cfg) in modules.items()},
            "optimizers": {},
        }
        write_manifest = lambda: (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        write_manifest()
        print(f"Benchmark folder: {out.resolve()}", flush=True)
        print(f"GPU: {manifest['environment']['gpu']}, torch {torch.__version__}", flush=True)

        data = ekf.load_data()

        manifest["validation"] = validate(data, modules, out / "validation", budget["population_size"],
                                          args.validate_workers)
        write_manifest()
        if not manifest["validation"]["passed"]:
            raise SystemExit("GPU validation failed: see validation/validation_table.csv. Benchmark not run.")
        if args.validate_only:
            return

        manifest["baseline"] = run_baseline(data, out / "baseline")
        write_manifest()

        for name in ORDER:
            module, cfg = modules[name]
            t0 = time.perf_counter()
            started = datetime.now().isoformat(timespec="seconds")
            print(f"\n################ {name}: seeds {list(budget['seeds'])}, population "
                  f"{budget['population_size']}, {budget['max_evals']} evaluations per seed ################",
                  flush=True)
            module.run_experiment(max_evals=budget["max_evals"], seeds=budget["seeds"], out_dir=out / name,
                                  population_size=budget["population_size"], n_workers=1, backend="gpu",
                                  data=data)
            checks = check_runs(out / name, name, budget["seeds"], budget["max_evals"], budget["population_size"])
            manifest["optimizers"][name] = {"started": started,
                                            "finished": datetime.now().isoformat(timespec="seconds"),
                                            "wall_clock_s": time.perf_counter() - t0, "checks": checks}
            write_manifest()
            if not all(c["ok"] for c in checks):
                raise SystemExit(f"{name}: evaluation accounting check failed; see manifest.json.")

        import plot_response
        plot_response.main(out)
        manifest["finished"] = datetime.now().isoformat(timespec="seconds")
        write_manifest()
        print(f"\nBenchmark complete: {out.resolve()}", flush=True)
    finally:
        if isinstance(sys.stdout, Tee):
            sys.stdout.file.close()
            sys.stdout = sys.stdout.stdout
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
