"""Everything in one command: prepare the datasets, run every optimizer on each, build the
tables and graphs.

    python main.py                 # the full benchmark (GPU), resumable
    python main.py --plan          # prepare the datasets, print the plan and a time estimate
    python main.py --plots-only    # rebuild tables and figures from results/<scenario>/
    python main.py --smoke         # tiny budget into results_smoke/<scenario>/, checks the pipeline
    python main.py --scenario gps16m   # the saved first run (any of the commands above)

Scenarios (SCENARIOS below), each on eth, inavwhu/playground00 and inavwhu/street00, results in
results/<scenario>/:
    outages   (default) the raw GPS, except three sections of 250 ft (76.2 m) of travel at 25, 50 and
              75 % of the path: about 5 m added error, about 15 m, and a complete outage. Also writes
              sections.md/png (plot_sections.py): how each filter behaves in those sections.
    gps16m    the first run, kept in results/gps16m/: 16 m white GPS noise everywhere, position-only.
Objective (NIS_WEIGHT): outages scores accuracy and consistency together, J = RMSE_h * (1 + |log10(mean
NIS / dof)|) (ekf.tuning.objective_value); gps16m keeps run 1's RMSE_h alone. GPS plausibility rule
(PLAUSIBILITY): outages discards any fix the platform could not have reached (ekf.filter.OPTIONS).
Preparing the datasets processes raw_data/ if needed and writes the degraded copies
(make_noisy_gps.py); both are skipped when the files exist. Optimizers, one after another so the GPU
serves one at a time: GA, PSO, DE, GWO, each with its own config (population 100, 1,000
evaluations per seed, seeds 1-5, BACKEND = "gpu"). Per dataset:

    1. validation   the first candidates GA scores for seed 1 are scored again by the filterpy
                    reference on the CPU; the run stops if any difference (J, RMSE, NIS) exceeds the tolerance
    2. baseline     THETA0 through the same fitness
    3. GA, PSO, DE, GWO: every seed, then the accounting check (n_evals = n_objective_evals =
                    convergence rows = MAX_EVALS) and the mean NIS of every incumbent on the
                    GPU (ekf.response), kept as data
    4. the dataset's table and convergence figure

and finally results/summary/: the table over all datasets and the summary figures (one panel per
dataset): convergence, and accuracy against convergence speed; and an OpenStreetMap map per dataset
(plot_maps.py) with the ground truth, the GPS, the baseline EKF and each optimizer's best run.

Resumable: a seed whose runs.csv row, convergence curve and incumbents file are complete is
skipped, so the command can be stopped and started again. Every session is appended to
results/<scenario>/main_manifest.json and everything printed goes to results/<scenario>/main.log.
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
import plot_maps
import plot_response
import plot_sections
from make_noisy_gps import SECTION_M, SECTIONS, derived_id, make, make_sections, sections_id

SOURCES = ["eth", "inavwhu/playground00", "inavwhu/street00"]
# Scenario gps16m: white GPS position noise per dataset (m, horizontal; vertical 1.5x, seed 0): 16 m, a raw
# GPS error of about 18.5 m (a poor receiver). Every dataset is position-only: iNav-WHU's F9P has no
# velocity, and ETH's velocity is dropped (with it, the clean velocity carries the filter and tuning
# gains < 1 %). (README, "GPS noise").
NOISE = {source: 16.0 for source in SOURCES}
DROP_VELOCITY = {"eth"}
# Scenario outages: raw GPS with the make_noisy_gps.SECTIONS degraded (5 m, 15 m, outage), seed 0.
SCENARIOS = {
    "outages": [sections_id(source) for source in SOURCES],
    "gps16m": [derived_id(source, NOISE[source], 0.0, 60.0, 0, source in DROP_VELOCITY) for source in SOURCES],
}
SCENARIO = "outages"
DATASETS = SCENARIOS[SCENARIO]
# Weight of NIS consistency in the objective per scenario (ekf.tuning.OBJECTIVE): 0 = RMSE_h alone.
NIS_WEIGHT = {"outages": 1.0, "gps16m": 0.0}
# GPS plausibility rule per scenario (ekf.filter.OPTIONS["plausibility"]: discard a fix the platform could
# not have reached since the last accepted one). Off for gps16m, which ran without it.
PLAUSIBILITY = {"outages": True, "gps16m": False}
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
        raise SystemExit("main.py needs BACKEND = 'gpu' in every config")


def results_root(scenario, smoke=False):
    return Path("results_smoke" if smoke else "results") / scenario


def _meta(dataset):
    path = ekf.PROCESSED_DIR / dataset / "meta.json"
    return json.loads(path.read_text()) if path.is_file() else {}


def prepare(datasets):
    """Make sure every dataset is processed: raw_data/ -> processed_data/, then the degraded copy.

    A copy is written again when its recorded settings differ from the ones here (the folder name does
    not show them all). Making a copy processes its source from raw_data/ first if needed."""
    for source in SOURCES:
        drop = source in DROP_VELOCITY
        noisy = derived_id(source, NOISE[source], 0.0, 60.0, 0, drop)
        if noisy in datasets:
            noise = _meta(noisy).get("gps_noise", {})
            if (noise.get("sigma_h_m"), noise.get("sigma_vel_ms"), noise.get("gm_sigma_m"), noise.get("seed"),
                    noise.get("velocity_dropped", False)) != (NOISE[source], 0.0, 0.0, 0, drop):
                make(source, NOISE[source], drop_vel=drop)
        if sections_id(source) in datasets:
            got = _meta(sections_id(source)).get("gps_sections", {})
            have = [{k: s.get(k) for k in want} for s, want in zip(got.get("sections", []), SECTIONS)]
            if (got.get("length_m"), got.get("seed"), have) != (SECTION_M, 0, list(SECTIONS)):
                make_sections(source)
    for ds in datasets:
        if not (ekf.PROCESSED_DIR / ds / "meta.json").is_file():
            ekf.load_data(ds)                           # a clean dataset: processed from raw_data/


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
            say(f"  validation: already passed (max difference of J {summary['max_diff_fitness']:.2e})")
            return summary
    out.mkdir(parents=True, exist_ok=True)
    n = min(n_candidates, population)
    say(f"  validation: GA seed 1, the first {n} candidates of generation 1, GPU vs filterpy ({n_workers} CPU workers)")
    with ekf.make_cost_fn(data, max_evals=n, backend="gpu") as tracker:
        GA.run_ga(data, tracker, seed=1, population_size=n)   # the same first n draws as the full population
        hist = tracker.convergence()
    thetas = hist[ekf.THETA_NAMES].to_numpy()
    table, summary = compare_with_cpu(data, thetas, n_workers=n_workers)
    table.insert(0, "tracker_fitness", hist["fitness"].to_numpy())
    table["diff_tracker_vs_cpu"] = np.abs(table["tracker_fitness"] - table["cpu_fitness"])
    scaled = table["diff_tracker_vs_cpu"] / np.maximum(1.0, table["cpu_fitness"].abs())
    summary["max_diff_tracker_vs_cpu"] = float(scaled[table["stable"]].max())
    summary["passed"] = bool(summary["passed"] and summary["max_diff_tracker_vs_cpu"] <= TOL_RMSE)
    summary.update(optimizer="GA", seed=1, generation=1)
    table.to_csv(out / "validation_table.csv", index=False)
    summary_path.write_text(json.dumps(summary, indent=2))
    if summary["n_unstable_excluded"]:
        say(f"  validation: {summary['n_unstable_excluded']} numerically chaotic candidate(s) (mean NIS > "
            f"{summary['nis_unstable']:g}) left out of the tolerance check; their |GPU - CPU| is up to "
            f"{summary['max_diff_fitness_unstable']:.1f} m, as rounding decides them on any backend")
    say(f"  validation: max |GPU - CPU| of J {summary['max_diff_fitness']:.2e} -> "
        f"{'PASSED' if summary['passed'] else 'FAILED'} (tolerance {TOL_RMSE:g}, relative above 1)")
    return summary


def baseline(data, out):
    if (out / "runs.csv").exists():
        r = pd.read_csv(out / "runs.csv").iloc[0]
        say(f"  baseline: already done (RMSE_h {r['rmse_h']:.4f} m, raw GPS {r['rmse_gps']:.4f} m)")
        return {"rmse_h": float(r["rmse_h"]), "rmse_gps": float(r["rmse_gps"])}
    rmse_gps = ekf.gps_rmse(data)
    with ekf.make_cost_fn(data, max_evals=1) as tracker:
        tracker(ekf.THETA0)
        row, _ = ekf.run_summary(data, tracker, "baseline", 0, rmse_gps=rmse_gps, population_size=1)
        rmse_base = row["rmse_h"]
        row.update(delta_vs_baseline=0.0, improvement_vs_baseline_pct=0.0)
        ekf.save_run(out, row, tracker)
    say(f"  baseline: THETA0 J {row['fitness']:.4f}, RMSE_h {rmse_base:.4f} m, mean NIS {row['mean_nis']:.2f} "
        f"(consistent: {row['nis_dof']}), raw GPS {rmse_gps:.4f} m")
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
        say(f"  mean NIS of {n_inc} incumbents (seeds {need}) on the GPU")
        for s, table in response.incumbent_nis(data, curves).items():
            table.to_csv(out / f"incumbents_seed{s}.csv", index=False)
    t_inc = time.perf_counter() - t1
    best = runs.loc[runs["fitness"].idxmin()]
    say(f"  {name} done: best seed {int(best['seed'])} J {best['fitness']:.4f}, RMSE_h {best['rmse_h']:.4f} m, "
        f"mean NIS {best['mean_nis']:.2f}; "
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
        extra = 0.05 * search                       # baseline, validation, re-scoring, incumbents' NIS
        total += search + extra
        say(f"  {ds:<22} {n_pred:>7,} predictions + {n_upd:>6,} GPS updates per candidate "
            f"(~{per_cand:.2f} s on the GPU); {evals:,} evaluations -> ~{(search + extra) / 3600:.1f} h")
    say(f"  total ~{total / 3600:.1f} h (an estimate, not a measurement)")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default=SCENARIO, choices=list(SCENARIOS),
                   help=f"which datasets and results/<scenario>/ (default {SCENARIO})")
    p.add_argument("--datasets", nargs="+", help="default: the scenario's datasets")
    p.add_argument("--optimizers", nargs="+", default=OPTIMIZERS, choices=OPTIMIZERS)
    p.add_argument("--plan", action="store_true", help="print the plan and a time estimate, run nothing")
    p.add_argument("--plots-only", action="store_true", help="only rebuild tables and figures from the results")
    p.add_argument("--smoke", action="store_true", help="tiny budget into results_smoke/")
    p.add_argument("--validate-candidates", type=int, default=24,
                   help="candidates in each dataset's GPU vs filterpy check")
    p.add_argument("--validate-workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    p.add_argument("--allow-any-branch", action="store_true")
    a = p.parse_args()
    a.datasets = a.datasets or SCENARIOS[a.scenario]
    ekf.tuning.OBJECTIVE["nis_weight"] = NIS_WEIGHT[a.scenario]
    ekf.filter.OPTIONS["plausibility"] = PLAUSIBILITY[a.scenario]

    if not a.smoke:
        check_configs()
    if not a.plots_only:
        prepare(a.datasets)
    if a.plan:
        plan(a.datasets, a.smoke)
        return
    root = results_root(a.scenario, a.smoke)
    if a.plots_only:
        plot_response.build_all(root, a.datasets, a.optimizers)
        plot_maps.main(a.datasets, root)
        plot_sections.main(a.datasets, root)
        return

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main" and not a.allow_any_branch:
        raise SystemExit(f"main.py runs from the main branch (it needs the data, the filter and all four optimizers); "
                         f"current branch is {branch!r}. Check out main or pass --allow-any-branch.")
    import torch
    from ekf import gpu
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to PyTorch; main.py uses the GPU backend.")

    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".main.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another main.py is running (delete it if none is).")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    sys.stdout = Tee(root / "main.log")
    session = {"started": datetime.now().isoformat(timespec="seconds"), "scenario": a.scenario, "smoke": a.smoke,
               "datasets": a.datasets, "optimizers": a.optimizers,
               "git": {"branch": branch, "commit": git("rev-parse", "HEAD"),
                       "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "torch": torch.__version__, "cuda": torch.version.cuda,
                               "gpu": torch.cuda.get_device_name(0)},
               "filter_options": dict(ekf.filter.OPTIONS), "objective": dict(ekf.tuning.OBJECTIVE),
               "configs": {o: {k: getattr(config(o), k) for k in dir(config(o)) if k.isupper()} for o in a.optimizers},
               "budget_override": SMOKE if a.smoke else None, "datasets_done": {}}
    manifest_path = root / "main_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"sessions": []}
    manifest["sessions"].append(session)

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    save_manifest()
    t_all = time.perf_counter()
    try:
        say(f"\n######## main.py {session['started']} | {a.scenario}: {', '.join(a.datasets)} | {', '.join(a.optimizers)} | "
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
            info = {"theta0": dict(zip(ekf.THETA_NAMES, data["theta0"].tolist())),
                    "validation": val, "baseline": baseline(data, out / "baseline"), "optimizers": {}}
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
        plot_maps.main(a.datasets, root)
        plot_sections.main(a.datasets, root)
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
