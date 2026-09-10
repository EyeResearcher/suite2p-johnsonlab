"""Run BatchProcessor from a YAML config file.

Usage:
    python scripts/run_batch.py --root /data/recordings --config config/batch_processing.yaml
    python scripts/run_batch.py --root /data/recordings  # uses config/batch_processing.yaml
    python scripts/run_batch.py --root /data/recordings --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite2p.still_cellpose import StillProcessor
from suite2p.still_cellpose_batch import BatchProcessor


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def batch_processor_from_config(cfg: dict, dry_run: bool = False) -> BatchProcessor:
    """Build a BatchProcessor from the structured config dict."""
    cellpose = cfg.get("cellpose", {})
    channels = cfg.get("channels", {})
    align = cfg.get("alignment", {})
    acq = cfg.get("acquisition", {})
    batch = cfg.get("batch", {})

    # Resolve the Cellpose model — fetches from HuggingFace if not a local file.
    model = StillProcessor.resolve_model(
        cellpose["model_name_or_path"],
        hf_repo_id=cellpose.get("hf_repo_id") or None,
    )

    # Load suite2p settings from a pre-validated .npy file or inline YAML block.
    settings_file = cfg.get("settings_file")
    if settings_file:
        import numpy as np
        p = Path(settings_file)
        # bare name → look in config/settings_defaults/ relative to repo root
        if not p.is_absolute() and len(p.parts) == 1:
            p = Path(__file__).resolve().parents[1] / "config" / "settings_defaults" / p
        if p.suffix != ".npy":
            p = p.with_suffix(".npy")
        suite2p_settings = np.load(p, allow_pickle=True).item()
    else:
        s2p = cfg.get("suite2p", {})
        suite2p_settings = {k: v for k, v in s2p.items() if v is not None} or None
    db_parameters = {k: v for k, v in acq.items() if v is not None}

    return BatchProcessor(
        model_name_or_path=model,
        diameter=cellpose.get("diameter"),
        cellprob_threshold=float(cellpose.get("cellprob_threshold", 0.0)),
        flow_threshold=float(cellpose.get("flow_threshold", 0.4)),
        still_channel=int(channels.get("still_channel", 1)),
        alignment_channel=channels.get("alignment_channel"),
        channel_axis=channels.get("channel_axis"),
        alignment_mode=str(align.get("mode", "translation")),
        auto_align=bool(align.get("auto_align", False)),
        alignment_min_response=float(align.get("min_response", 0.1)),
        alignment_min_ecc=float(align.get("min_ecc", 0.5)),
        alignment_max_abs_shift=float(align.get("max_abs_shift", 50.0)),
        dy=int(align.get("dy", 0)),
        dx=int(align.get("dx", 0)),
        suite2p_settings=suite2p_settings or None,
        db_parameters=db_parameters or None,
        output_folder=str(batch.get("output_folder", "suite2p_still")),
        skip_completed=bool(batch.get("skip_completed", True)),
        save_qc=bool(batch.get("save_qc", True)),
        delete_binary_after=bool(batch.get("delete_binary_after", True)),
        stop_on_error=bool(batch.get("stop_on_error", False)),
        dry_run=bool(batch.get("dry_run", False)) or dry_run,
        device=batch.get("device") or None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Suite2p still-Cellpose processing.")
    parser.add_argument("--root", required=True, type=Path, help="Root folder containing recordings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "batch_processing.yaml",
        help="Path to batch_processing.yaml (defaults to config/batch_processing.yaml).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover and plan without processing.")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    proc = batch_processor_from_config(cfg, dry_run=args.dry_run)

    print(f"Root:   {args.root}")
    print(f"Config: {args.config}")
    print(f"Model:  {proc.model_name_or_path}")
    print()

    results = proc.run(args.root)
    print()

    report_path = args.root / "still_cellpose_batch_results.json"
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n}")

    print(f"\nFull results written to {report_path}")

    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\n{len(failed)} failure(s):")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', '')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
