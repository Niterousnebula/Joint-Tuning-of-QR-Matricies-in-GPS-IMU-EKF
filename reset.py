"""Remove the generated data so the next `python main.py` starts from scratch.

    python reset.py            # list what would be removed, then ask before removing it
    python reset.py --yes      # remove without asking

Removes results/ and results_smoke/ (every run, table and figure), the noisy datasets written by
make_noisy_gps.py (processed_data/ folders whose meta.json has "derived_from"; main.py writes them again)
and .tile_cache/ (map tiles). Keeps raw_data/, the clean processed datasets and all code.
Refuses while a main.py run holds results/.main.lock.
"""
import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def targets():
    found = [ROOT / name for name in ("results", "results_smoke", ".tile_cache") if (ROOT / name).exists()]
    for meta in sorted((ROOT / "processed_data").glob("**/meta.json")):
        if "derived_from" in json.loads(meta.read_text()):
            found.append(meta.parent)
    return found


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--yes", action="store_true", help="remove without asking")
    a = p.parse_args()
    if (ROOT / "results" / ".main.lock").exists():
        raise SystemExit("results/.main.lock exists: a main.py run is in progress (delete the lock if none is).")
    found = targets()
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
