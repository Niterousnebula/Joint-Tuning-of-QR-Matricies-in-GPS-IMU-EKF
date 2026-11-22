"""Remove the generated data of one scenario so the next `python main.py` starts it from scratch.

    python reset.py                    # the current scenario (main.SCENARIO): list, then ask before removing
    python reset.py --scenario gps16m  # the saved first run instead
    python reset.py --all              # every scenario, results_smoke/, every derived dataset and the tile cache
    python reset.py --yes              # remove without asking

For a scenario it removes results/<scenario>/ and results_smoke/<scenario>/ (every run, table and figure)
and the scenario's degraded datasets in processed_data/ (main.py writes them again). Other scenarios'
results are kept. Keeps raw_data/, the clean processed datasets and all code.
Refuses while a main.py run holds a .main.lock in what it would remove.
"""
import argparse
import json
import shutil
from pathlib import Path

import main as runner

ROOT = Path(__file__).resolve().parent


def targets(scenario):
    if scenario is None:
        found = [ROOT / name for name in ("results", "results_smoke", ".tile_cache")]
        for meta in sorted((ROOT / "processed_data").glob("**/meta.json")):
            if "derived_from" in json.loads(meta.read_text()):
                found.append(meta.parent)
    else:
        found = [ROOT / runner.results_root(scenario), ROOT / runner.results_root(scenario, smoke=True)]
        found += [ROOT / "processed_data" / ds for ds in runner.SCENARIOS[scenario]]
    return [t for t in found if t.exists()]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scenario", default=runner.SCENARIO, choices=list(runner.SCENARIOS))
    p.add_argument("--all", action="store_true", help="every scenario, every derived dataset and the tile cache")
    p.add_argument("--yes", action="store_true", help="remove without asking")
    a = p.parse_args()
    found = targets(None if a.all else a.scenario)
    locks = [lock for t in found for lock in t.glob("**/.main.lock")]
    if locks:
        raise SystemExit(f"{locks[0].relative_to(ROOT)} exists: a main.py run is in progress (delete the lock if none is).")
    if not found:
        print("Nothing to remove.")
        return
    print("Will remove:")
    for t in found:
        print(f"  {t.relative_to(ROOT)}")
    if not a.yes and input("Remove these? [y/N] ").strip().lower() != "y":
        print("Nothing removed.")
        return
    for t in found:
        shutil.rmtree(t)
    print(f"Removed {len(found)} folders. Run `python main.py` to regenerate the datasets and results.")


if __name__ == "__main__":
    main()
