"""Run BatchProcessor on a clean copy of one recording and compare against reference."""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from suite2p.still_cellpose_batch import BatchProcessor

ROOT = Path(r"C:\Users\mzinn1\Desktop\Suite2pTest_run")
REF  = Path(r"C:\Users\mzinn1\Desktop\Suite2pTest\Region_1\Day_10\10-1_BP\suite2p\plane0")

proc = BatchProcessor(
    model_name_or_path=r"C:\Users\mzinn1\Desktop\CellposeTrainingDatasets\InVitroRGC_Snaps\models\cpdino_RGC-Snap",
    still_channel=1,
    alignment_channel=1,
    alignment_mode="auto",
    auto_align=True,
    suite2p_settings={
        "fs": 15.0,
        "tau": 1.0,
        "diameter": [12.0, 12.0],
        "torch_device": "cuda",
    },
    db_parameters={
        "nchannels": 1,
        "functional_chan": 1,
    },
    output_folder="suite2p",
    save_qc=True,
    delete_binary_after=True,
)

print("Running BatchProcessor on clean folder...")
results = proc.run(ROOT)
print()
for r in results:
    print(f"  {r['status']}  {r['name']}  n_masks={r.get('n_masks')}  n_rois={r.get('n_rois')}")

assert len(results) == 1 and results[0]["status"] == "processed", f"Unexpected result: {results}"
out = Path(results[0]["output"])

print("\n-- Comparing outputs against reference --")

# Masks — Cellpose is non-deterministic; expect high but not 100% agreement
ref_masks = np.load(REF / "cellpose_masks.npy")
new_masks = np.load(out / "cellpose_masks.npy")
mask_pct  = np.mean(new_masks == ref_masks) * 100
print(f"cellpose_masks pixel agreement: {mask_pct:.2f}%  (non-determinism expected)")

ref_hi = np.load(REF / "cellpose_masks_hires.npy")
new_hi = np.load(out / "cellpose_masks_hires.npy")
hi_pct = np.mean(new_hi == ref_hi) * 100
print(f"cellpose_masks_hires pixel agreement: {hi_pct:.2f}%")

# ROI count — allow small variance from Cellpose non-determinism
ref_stat = np.load(REF / "stat.npy", allow_pickle=True)
new_stat = np.load(out / "stat.npy", allow_pickle=True)
roi_diff_pct = abs(len(new_stat) - len(ref_stat)) / len(ref_stat) * 100
print(f"stat.npy: {len(new_stat)} ROIs  (reference: {len(ref_stat)},  diff: {roi_diff_pct:.1f}%)")
assert roi_diff_pct < 10, f"ROI count differs by {roi_diff_pct:.1f}% — unexpectedly large"

# Fluorescence traces — shape and rough correlation
ref_F = np.load(REF / "F.npy")
new_F = np.load(out / "F.npy")
n = min(len(ref_F), len(new_F))
corr = float(np.corrcoef(ref_F[:n].ravel(), new_F[:n].ravel())[0, 1])
print(f"F.npy: {new_F.shape}  (reference: {ref_F.shape})  trace correlation={corr:.4f}")
assert new_F.shape[1] == ref_F.shape[1], "Frame count mismatch"
assert mask_pct > 70, f"Pixel agreement {mask_pct:.1f}% is unexpectedly low"

print("\nAll comparisons complete.")
