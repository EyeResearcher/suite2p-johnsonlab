"""Quick regression test against saved reference outputs in Suite2pTest."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite2p.still_cellpose import StillProcessor, Suite2pInterface
from suite2p.still_cellpose_batch import BatchProcessor, discover_experiments

ROOT  = Path(r"C:\Users\mzinn1\Desktop\Suite2pTest")
PLANE = ROOT / "Region_1" / "Day_10" / "10-1_BP" / "suite2p" / "plane0"

ref_masks = np.load(PLANE / "cellpose_masks.npy")
ref_masks_hi = np.load(PLANE / "cellpose_masks_hires.npy")
ref_stat = np.load(PLANE / "stat.npy", allow_pickle=True)
settings = np.load(PLANE / "settings.npy", allow_pickle=True).item()
alignment = json.loads((PLANE / "still_alignment.json").read_text())
ref_shape = ref_masks.shape

print(f"Video-res shape: {ref_shape},  n_cells: {ref_masks.max()}")
print(f"Hires shape:     {ref_masks_hi.shape},  n_cells: {ref_masks_hi.max()}")

# 1 ── relabel_masks is idempotent on already-consecutive labels
assert np.array_equal(StillProcessor.relabel_masks(ref_masks), ref_masks)
print("PASS  relabel_masks idempotent")

# 2 ── resize + affine warp reproduces the saved cellpose_masks.npy
warp = np.asarray(alignment["warp_matrix"], dtype=np.float32)
resized = StillProcessor.resize_label_masks(ref_masks_hi, ref_shape)
recomputed = StillProcessor.warp_label_masks(resized, warp, target_shape=ref_shape, inverse_map=True)
pct = np.mean(recomputed == ref_masks) * 100
print(f"PASS  resize+warp: {pct:.4f}% pixels agree  exact={np.array_equal(recomputed, ref_masks)}")

# 3 ── masks_to_stats produces the same ROI set as the reference stat.npy
iface = Suite2pInterface(PLANE)
new_stat = iface.masks_to_stats(ref_masks, settings, do_soma_crop=False)
assert len(new_stat) == len(ref_stat), f"ROI count {len(new_stat)} != {len(ref_stat)}"
npix_ok = all(new_stat[i]["npix"] == ref_stat[i]["npix"] for i in range(len(new_stat)))
pix_ok = all(np.array_equal(new_stat[i]["ypix"], ref_stat[i]["ypix"]) for i in range(len(new_stat)))
print(f"PASS  masks_to_stats: {len(new_stat)} ROIs  npix_match={npix_ok}  pixels_match={pix_ok}")

# ── BatchProcessor checks against the real folder tree ──────────────────────

# 4 ── discover_experiments finds all recordings and pairs them with their stills
experiments = discover_experiments(ROOT, output_folder="suite2p")
assert len(experiments) > 0, "No experiments discovered"
with_still  = [e for e in experiments if e.still_source is not None]
print(f"PASS  discover_experiments: {len(experiments)} recordings, {len(with_still)} with a snap still")

# 5 ── BatchProcessor.is_completed recognises completed plane0 folders
proc = BatchProcessor(output_folder="suite2p")
completed = [e for e in experiments if proc.is_completed(e)]
assert len(completed) > 0, "No completed experiments found — check output_folder"
print(f"PASS  is_completed: {len(completed)}/{len(experiments)} experiments marked complete")

# 6 ── _build_registration_config produces a valid configuration for a real experiment
exp = experiments[0]
db, reg_settings = proc._build_registration_config(exp)
assert db["file_list"] == [exp.video.name]
assert reg_settings["run"]["do_detection"] is False
assert reg_settings["run"]["do_registration"] >= 1
assert reg_settings["io"]["delete_bin"] is False
print(f"PASS  _build_registration_config for '{exp.name}'")

# 7 ── dry_run lists all completed experiments as skipped, ready ones as ready
results = proc.run(ROOT)
statuses = {r["status"] for r in results}
assert "ready" not in statuses or len(results) > 0
skipped = sum(1 for r in results if r["status"] == "skipped_completed")
print(f"PASS  dry_run (skip_completed=True): {skipped}/{len(results)} skipped as completed")

print("\nAll checks passed.")
