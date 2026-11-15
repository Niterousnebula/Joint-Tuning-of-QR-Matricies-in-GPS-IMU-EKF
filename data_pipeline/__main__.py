"""Command line: list the datasets in raw_data/, pick one, and write it to processed_data/."""
import argparse
import json
import sys
from pathlib import Path

from . import DEFAULT_OUT, DEFAULT_RAW, discover, load, select, write


def _status(out_root, dataset_id):
    """'ready' / 'not ready' from the processed meta.json, or '-' if not processed yet."""
    path = Path(out_root) / dataset_id / "meta.json"
    if not path.is_file():
        return "-"
    return "ready" if json.loads(path.read_text()).get("ekf_ready", {}).get("ready") else "not ready"


def _print_list(cands, out_root, needle=None):
    print(f"      {'id':<58} {'EKF status':<10} description")
    for i, c in enumerate(cands, 1):
        if needle and needle.lower() not in c.id.lower() and needle.lower() not in c.label.lower():
            continue
        print(f"  {i:3d}. {c.id:<58} {_status(out_root, c.id):<10} {c.label}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m data_pipeline", description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW, help="raw data root (default: raw_data/)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root (default: processed_data/)")
    parser.add_argument("--list", action="store_true", help="only list the datasets")
    parser.add_argument("--select", metavar="NUMBER_OR_ID", help="choose a dataset without prompting")
    parser.add_argument("--filter", metavar="TEXT", help="only show datasets whose id or label contains TEXT")
    args = parser.parse_args(argv)

    cands = discover(args.raw)
    if not cands:
        print(f"no datasets found under {args.raw}", file=sys.stderr)
        return 1
    print(f"Datasets in {args.raw}:")
    _print_list(cands, args.out, args.filter)
    if args.list:
        return 0

    choice = args.select
    if choice is None:
        try:
            choice = input("\nSelect a dataset (number or id): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 1

    try:
        cand = select(cands, choice)
    except KeyError as err:
        print(f"error: {err.args[0]}", file=sys.stderr)
        return 2

    print(f"\nLoading {cand.id} ...")
    rec = load(cand.id, args.raw)
    out = write(rec, args.out)
    m = rec.meta
    print(f"  IMU   {m['imu_samples']} samples, {m['imu_duration_s']:.1f} s, "
          f"{m['imu_rate_hz']:.1f} Hz, max gap {m['imu_max_gap_s']:.3f} s, "
          f"invalid rows dropped {m['imu_dropped_invalid']}, "
          f"rows outside kept segment {m['imu_dropped_outside_kept_segment']}")
    print(f"  GPS   {m.get('gps_samples', 0)} fixes" if rec.gps is not None else "  GPS   none")
    print(f"  GT    {m.get('gt_samples', 0)} samples" if rec.gt is not None else "  GT    none")
    ready = m["ekf_ready"]
    print(f"  EKF   {'ready' if ready['ready'] else 'NOT ready'}")
    for reason in ready["reasons"]:
        print(f"          - {reason}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
