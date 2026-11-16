"""Run every optimizer on every dataset, then build the tables, graphs and 4D response surfaces.

    python run_all.py                 # the full benchmark (GPU), resumable
    python run_all.py --plan          # what would run, with a time estimate; runs nothing
    python run_all.py --plots-only    # rebuild tables and figures from results/
    python run_all.py --smoke         # tiny budget into results_smoke/, checks the pipeline

Datasets: eth, inavwhu/playground00, inavwhu/street00. Optimizers, one after another so the GPU
serves one at a time: GA, PSO, DE, GWO, each with its own config (population 200, 2,000
evaluations per seed, seeds 1-5, BACKEND = "gpu"). Per dataset:

    1. validation   the first candidates GA scores for seed 1 are scored again by the filterpy
                    reference on the CPU; the run stops if any difference exceeds TOL_RMSE
    2. baseline     THETA0 through the same fitness
    3. GA, PSO, DE, GWO: every seed, then the accounting check (n_evals = n_objective_evals =
                    convergence rows = MAX_EVALS) and the robustness of every incumbent on the
                    GPU (ekf.response), which the response surfaces need
    4. the dataset's table, convergence and robustness figures

and finally results/summary/: the table over all datasets and one 4D response surface per
optimizer (accuracy x robustness x convergence, coloured by compute).

Resumable: a seed whose runs.csv row, convergence curve and incumbents file are complete is
skipped, so the command can be stopped and started again. Every session is appended to
results/run_all_manifest.json and everything printed goes to results/run_all.log.
"""
import argparse
import importlib
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
import plot_response

DATASETS = ["eth", "inavwhu/playground00", "inavwhu/street00"]
OPTIMIZERS = ["GA", "PSO", "DE", "GWO"]
LIVE_LOGS = {"PSO": "pso_log.csv", "DE": "de_log.csv", "GWO": "gwo_log.csv"}   # GA keeps no live log
SMOKE = {"population_size": 4, "max_evals": 8, "seeds": (1,), "validate": 4}
STEP_SECONDS = 1.75e-6     # GPU s per candidate per EKF step at population 200 (RTX 4060, ETH: 0.071 s / 40,576 steps)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
class Tee:
    """Copy everything printed into the run log."""

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


def say(text=""):
    print(text, flush=True)


def config(name):
    return importlib.import_module(f"{name}_config")


def population_of(cfg):
    return getattr(cfg, "POPULATION_SIZE", getattr(cfg, "SWARM_SIZE", None))


def budget_of(name, smoke):
    cfg = config(name)
    if smoke:
        return SMOKE["population_size"], SMOKE["max_evals"], SMOKE["seeds"]
    return population_of(cfg), cfg.MAX_EVALS, tuple(cfg.SEEDS)


def check_configs():
    """Same budget and backend for every optimizer, or the comparison is not fair."""
    got = {name: (*budget_of(name, False), getattr(config(name), "BACKEND", None)) for name in OPTIMIZERS}
    if len(set(got.values())) != 1:
        raise SystemExit("The optimizer configs differ (population, max_evals, seeds, backend):\n  "
                         + "\n  ".join(f"{k}: {v}" for k, v in got.items()))
    if next(iter(got.values()))[3] != "gpu":
        raise SystemExit("run_all.py needs BACKEND = 'gpu' in every config")


def dataset_steps(dataset):
    """IMU predictions and GPS updates per candidate, from the processed files (no EKF run)."""
    folder = ekf.PROCESSED_DIR / dataset
    meta = json.loads((folder / "meta.json").read_text())
    return int(meta["imu_samples"]) - 1, int(meta["gps_samples"]) - 1


def seed_state(out, seed, max_evals, population):
    """'done', 'needs_incumbents' or 'run' for one seed of one optimizer."""
    runs = out / "runs.csv"
    conv = out / f"convergence_seed{seed}.csv"
    if not runs.exists() or not conv.exists():
        return "run"
    r = pd.read_csv(runs)
    r = r[r["seed"] == seed]
    if len(r) != 1:
        return "run"
    r = r.iloc[0]
    complete = (int(r["n_evals"]) == max_evals and int(r.get("n_objective_evals", -1)) == max_evals
                and int(r.get("population_size", -1)) == population
                and sum(1 for _ in open(conv, encoding="utf-8")) - 1 == max_evals)
    if not complete:
        return "run"
    return "done" if (out / f"incumbents_seed{seed}.csv").exists() else "needs_incumbents"


def drop_log_rows(path, seeds):
    """Remove a seed's rows from an append-only live log before that seed is run again."""
    if path.exists() and path.stat().st_size:
        log = pd.read_csv(path)
        log[~log["seed"].isin(seeds)].to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def validate(data, out, population, n_candidates, n_workers):
    """GA seed 1, first n_candidates of generation 1: optimizer -> tracker -> GPU vs filterpy."""
    from ekf.gpu import TOL_RMSE, compare_with_cpu
    import GA

    summary_path = out / "validation_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("passed"):
            say(f"  validation: already passed (max difference {summary['max_diff_fitness']:.2e} m)")
            return summary
    out.mkdir(parents=True, exist_ok=True)
    n = min(n_candidates, population)
    say(f"  validation: GA seed 1, the first {n} candidates of generation 1, GPU vs filterpy ({n_workers} CPU workers)")
    with ekf.make_cost_fn(data, max_evals=n, backend="gpu") as tracker:
        GA.run_ga(data, tracker, seed=1, population_size=n)   # the same first n draws as population 200
        hist = tracker.convergence()
    thetas = hist[ekf.THETA_NAMES].to_numpy()
    table, summary = compare_with_cpu(data, thetas, n_workers=n_workers)
    table.insert(0, "tracker_fitness", hist["fitness"].to_numpy())
    table["diff_tracker_vs_cpu"] = np.abs(table["tracker_fitness"] - table["cpu_rmse_h"])
    summary["max_diff_tracker_vs_cpu"] = float(table["diff_tracker_vs_cpu"].max())
    summary["passed"] = bool(summary["passed"] and summary["max_diff_tracker_vs_cpu"] <= TOL_RMSE)
    summary.update(optimizer="GA", seed=1, generation=1)
    table.to_csv(out / "validation_table.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    say(f"  validation: max |GPU - CPU| {summary['max_diff_fitness']:.2e} m -> "
        f"{'PASSED' if summary['passed'] else 'FAILED'} (tolerance {TOL_RMSE:g} m)")
    return summary


def baseline(data, out):
    if (out / "runs.csv").exists():
        r = pd.read_csv(out / "runs.csv").iloc[0]
        say(f"  baseline: already done (RMSE_h {r['rmse_h']:.4f} m, raw GPS {r['rmse_gps']:.4f} m)")
        return {"rmse_h": float(r["rmse_h"]), "rmse_gps": float(r["rmse_gps"])}
    rmse_gps = ekf.gps_rmse(data)
    with ekf.make_cost_fn(data, max_evals=1) as tracker:
        rmse_base = tracker(ekf.THETA0)
        row, _ = ekf.run_summary(data, tracker, "baseline", 0, rmse_gps=rmse_gps,
                                 rmse_baseline=rmse_base, population_size=1)
        ekf.save_run(out, row, tracker)
    say(f"  baseline: THETA0 RMSE_h {rmse_base:.4f} m, raw GPS {rmse_gps:.4f} m, mean NIS {row['mean_nis']:.2f}")
    return {"rmse_h": rmse_base, "rmse_gps": rmse_gps}


def run_optimizer(name, data, out, smoke):
    population, max_evals, seeds = budget_of(name, smoke)
    states = {s: seed_state(out, s, max_evals, population) for s in seeds}
    to_run = [s for s, st in states.items() if st == "run"]
    say(f"\n  ---- {name}: population {population}, {max_evals} evaluations x seeds {list(seeds)} ----")
    say("  seeds: " + ", ".join(f"{s} {st}" for s, st in states.items()))
    t0 = time.perf_counter()
    if to_run:
        if name in LIVE_LOGS:
            drop_log_rows(out / LIVE_LOGS[name], to_run)
        module = importlib.import_module(name)
        module.run_experiment(max_evals=max_evals, seeds=tuple(to_run), out_dir=out,
                              population_size=population, n_workers=1, backend="gpu", data=data)
    t_search = time.perf_counter() - t0

    checks = []
    runs = pd.read_csv(out / "runs.csv")
    for s in seeds:
        r = runs[runs["seed"] == s].iloc[0]
        n_curve = sum(1 for _ in open(out / f"convergence_seed{s}.csv", encoding="utf-8")) - 1
        ok = int(r["n_evals"]) == int(r["n_objective_evals"]) == n_curve == max_evals
        checks.append({"seed": s, "n_evals": int(r["n_evals"]), "n_objective_evals": int(r["n_objective_evals"]),
                       "convergence_rows": n_curve, "ok": bool(ok)})
        if not ok:
            raise SystemExit(f"{name} seed {s}: evaluation accounting failed ({checks[-1]})")

    need = [s for s in seeds if not (out / f"incumbents_seed{s}.csv").exists()]
    t1 = time.perf_counter()
    if need:
        curves = {s: pd.read_csv(out / f"convergence_seed{s}.csv") for s in need}
        from ekf import response
        n_inc = sum(len(response.incumbents(c)) for c in curves.values())
        say(f"  robustness of {n_inc} incumbents (seeds {need}) on the GPU, {len(ekf.PERTURBATIONS)} perturbations")
        for s, table in response.incumbent_robustness(data, curves).items():
            table.to_csv(out / f"incumbents_seed{s}.csv", index=False)
    t_inc = time.perf_counter() - t1
    best = runs.loc[runs["rmse_h"].idxmin()]
    say(f"  {name} done: best seed {int(best['seed'])} RMSE_h {best['rmse_h']:.4f} m; "
        f"search {t_search / 60:.1f} min, incumbents {t_inc / 60:.1f} min")
    return {"seeds_run": to_run, "search_s": t_search, "incumbents_s": t_inc, "checks": checks}


def plan(datasets, smoke):
    say("Plan (estimate from the dataset sizes; GPU time scales with EKF steps per candidate):")
    total = 0.0
    for ds in datasets:
        n_pred, n_upd = dataset_steps(ds)
        per_cand = (n_pred + n_upd) * STEP_SECONDS
        evals = sum(budget_of(o, smoke)[1] * len(budget_of(o, smoke)[2]) for o in OPTIMIZERS)
        search = evals * per_cand
        extra = 0.20 * search                       # incumbents' robustness and re-scoring, roughly
        total += search + extra
        say(f"  {ds:<22} {n_pred:>7,} predictions + {n_upd:>6,} GPS updates per candidate "
            f"(~{per_cand:.2f} s on the GPU); {evals:,} evaluations -> ~{(search + extra) / 3600:.1f} h")
    say(f"  total ~{total / 3600:.1f} h (an estimate, not a measurement)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--optimizers", nargs="+", default=OPTIMIZERS, choices=OPTIMIZERS)
    p.add_argument("--plan", action="store_true", help="print the plan and a time estimate, run nothing")
    p.add_argument("--plots-only", action="store_true", help="only rebuild tables and figures from the results")
    p.add_argument("--smoke", action="store_true", help="tiny budget into results_smoke/")
    p.add_argument("--rescore-workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)),
                   help="CPU processes for the filterpy re-scoring of final thetas")
    p.add_argument("--validate-candidates", type=int, default=24,
                   help="candidates in each dataset's GPU vs filterpy check")
    p.add_argument("--validate-workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    p.add_argument("--allow-any-branch", action="store_true")
    a = p.parse_args()

    if not a.smoke:
        check_configs()
    if a.plan:
        plan(a.datasets, a.smoke)
        return
    root = Path("results_smoke" if a.smoke else "results")
    if a.plots_only:
        plot_response.build_all(root, a.datasets, a.optimizers)
        return

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main" and not a.allow_any_branch:
        raise SystemExit(f"run_all.py runs from main (it needs the data, the filter and all four optimizers); "
                         f"current branch is {branch!r}. Check out main or pass --allow-any-branch.")
    import torch
    from ekf import gpu
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch; run_all.py uses the GPU backend.")

    root.mkdir(exist_ok=True)
    lock = root / ".run_all.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another run_all.py is running (delete it if none is).")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    sys.stdout = Tee(root / "run_all.log")
    ekf.tuning.RESCORE_WORKERS = a.rescore_workers
    session = {"started": datetime.now().isoformat(timespec="seconds"), "smoke": a.smoke,
               "datasets": a.datasets, "optimizers": a.optimizers,
               "git": {"branch": branch, "commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0)},
               "filter_options": dict(ekf.filter.OPTIONS),
               "configs": {o: {k: getattr(config(o), k) for k in dir(config(o)) if k.isupper()} for o in a.optimizers},
               "budget_override": SMOKE if a.smoke else None, "datasets_done": {}}
    manifest_path = root / "run_all_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": []}
    manifest["sessions"].append(session)

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    save_manifest()
    t_all = time.perf_counter()
    try:
        say(f"\n######## run_all {session['started']} | {', '.join(a.datasets)} | {', '.join(a.optimizers)} | "
            f"GPU {session['environment']['gpu']} ########")
        for ds in a.datasets:
            t_ds = time.perf_counter()
            n_pred, n_upd = dataset_steps(ds)
            say(f"\n==================== {ds}: {n_pred:,} predictions + {n_upd:,} GPS updates per candidate "
                f"====================")
            data = ekf.load_data(ds)
            out = root / ds
            val = validate(data, out / "validation", budget_of("GA", a.smoke)[0],
                           SMOKE["validate"] if a.smoke else a.validate_candidates, a.validate_workers)
            if not val["passed"]:
                raise SystemExit(f"{ds}: GPU vs filterpy validation failed; see {out / 'validation'}")
            info = {"validation": val, "baseline": baseline(data, out / "baseline"), "optimizers": {}}
            for name in a.optimizers:
                info["optimizers"][name] = run_optimizer(name, data, out / name, a.smoke)
                session["datasets_done"][ds] = info
                save_manifest()
            plot_response.dataset_figures(out, ds, a.optimizers)
            info["wall_clock_s"] = time.perf_counter() - t_ds
            save_manifest()
            gpu.clear_cache()
            say(f"  {ds} finished in {info['wall_clock_s'] / 3600:.2f} h")
        plot_response.summary(root, a.datasets, a.optimizers)
        session["finished"] = datetime.now().isoformat(timespec="seconds")
        session["wall_clock_s"] = time.perf_counter() - t_all
        save_manifest()
        say(f"\nAll done in {session['wall_clock_s'] / 3600:.2f} h. Summary: {root / 'summary'}")
    finally:
        if isinstance(sys.stdout, Tee):
            sys.stdout.file.close()
            sys.stdout = sys.stdout.stdout
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
