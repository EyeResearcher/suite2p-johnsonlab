"""Batch processing for Suite2p extraction using still-image Cellpose masks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Collection, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .parameters import default_db, default_settings
from .run_s2p import run_s2p
from .still_cellpose import StillProcessor, Suite2pInterface


TIFF_EXTENSIONS = {".tif", ".tiff"}
StillLoader = Callable[
    [Path, int, int | None, Mapping[str, int] | None], np.ndarray
]


@dataclass(frozen=True)
class Experiment:
    """Input and output paths associated with one recording."""

    name: str
    folder: Path
    video: Path
    still_tiff: Path | None
    still_oir: Path | None
    output_root: Path

    @property
    def plane_path(self) -> Path:
        return self.output_root / "plane0"

    @property
    def still_source(self) -> Path | None:
        return self.still_tiff or self.still_oir


def _is_generated_path(path: Path) -> bool:
    generated_parts = {"suite2p", "suite2p_still", "metrics", "reg_tif"}
    return any(part.lower() in generated_parts for part in path.parts)


def _video_candidates(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TIFF_EXTENSIONS
        and not path.stem.lower().endswith("_snap")
        and not _is_generated_path(path.relative_to(root))
    )


def _find_stills(folder: Path, video_stem: str) -> tuple[Path | None, Path | None]:
    candidates = [
        path for path in folder.rglob(f"{video_stem}_snap.*")
        if path.is_file() and not _is_generated_path(path.relative_to(folder))
    ]
    tiffs = sorted(path for path in candidates if path.suffix.lower() in TIFF_EXTENSIONS)
    oirs = sorted(path for path in candidates if path.suffix.lower() == ".oir")
    return (tiffs[0] if tiffs else None, oirs[0] if oirs else None)


def discover_experiments(
    root: str | Path,
    output_folder: str = "suite2p_still",
) -> list[Experiment]:
    """Discover recording TIFFs and their matching ``*_snap`` stills."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    experiments = []
    for video in _video_candidates(root):
        still_tiff, still_oir = _find_stills(video.parent, video.stem)
        experiments.append(
            Experiment(
                name=video.stem,
                folder=video.parent,
                video=video,
                still_tiff=still_tiff,
                still_oir=still_oir,
                output_root=video.parent / output_folder,
            )
        )
    return experiments


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge nested pipeline parameters into a settings dictionary."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


@dataclass
class BatchProcessor:
    """Configuration and orchestration for batch still-Cellpose Suite2p processing."""

    # Segmentation
    model_name_or_path: str = "cpsam"
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4

    # Channels
    still_channel: int = 1
    alignment_channel: int | None = None
    channel_axis: int | None = None

    # Alignment
    alignment_mode: str = "translation"
    auto_align: bool = False
    alignment_min_response: float = 0.1
    alignment_min_ecc: float = 0.5
    alignment_max_abs_shift: float = 50.0
    dy: int = 0
    dx: int = 0

    # OIR conversion
    oir_axis_indices: Mapping[str, int] | None = None
    oir_environment: str = "image_conversion"
    oir_converter_script: str | Path | None = None

    # Suite2p configuration
    suite2p_settings: Mapping[str, Any] | None = None
    db_parameters: Mapping[str, Any] | None = None

    # Batch control
    output_folder: str = "suite2p_still"
    still_loader: StillLoader | None = None
    exclude_names: Collection[str] | None = None
    skip_completed: bool = True
    save_qc: bool = True
    delete_binary_after: bool = True
    stop_on_error: bool = False
    dry_run: bool = False
    device: str | None = None

    def _build_registration_config(
        self, experiment: Experiment
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a registration-only Suite2p configuration for one experiment."""
        settings = default_settings()
        deep_merge(settings, self.suite2p_settings or {})
        settings["run"]["do_registration"] = max(1, int(settings["run"]["do_registration"]))
        settings["run"]["do_detection"] = False
        settings["io"]["delete_bin"] = False
        settings["registration"]["reg_tif"] = False
        settings["registration"]["reg_tif_chan2"] = False

        db = default_db()
        deep_merge(db, self.db_parameters or {})
        db["data_path"] = [str(experiment.folder)]
        db["file_list"] = [experiment.video.name]
        db["save_path0"] = str(experiment.folder)
        db["save_folder"] = experiment.output_root.name
        db["fast_disk"] = str(experiment.output_root)
        db["input_format"] = "tif"
        db["look_one_level_down"] = False
        db["subfolders"] = None
        return db, settings

    @staticmethod
    def is_completed(experiment: Experiment) -> bool:
        """Return whether all principal still-mask outputs exist."""
        required = ("F.npy", "Fneu.npy", "spks.npy", "stat.npy", "iscell.npy", "ops.npy")
        return all((experiment.plane_path / name).is_file() for name in required)

    def _load_still_channel(self, path: Path) -> np.ndarray:
        """Load a TIFF still at the instance's channel settings."""
        if path.suffix.lower() == ".oir":
            raise ValueError("OIR files must pass through the conversion environment")
        return StillProcessor(path=path, channel=self.still_channel, channel_axis=self.channel_axis).load()

    def _load_segmentation_still(self, still_source: Path) -> tuple[np.ndarray, Path | None]:
        """Return (still_array, converted_tiff_or_None) for the segmentation channel."""
        if self.still_loader is not None:
            return self.still_loader(
                still_source, self.still_channel, self.channel_axis, self.oir_axis_indices
            ), None
        if still_source.suffix.lower() == ".oir":
            converted = convert_oir_in_environment(
                still_source,
                channel=self.still_channel,
                axis_indices=self.oir_axis_indices,
                environment=self.oir_environment,
                converter_script=self.oir_converter_script,
            )
            return StillProcessor(path=converted, channel=0).load(), converted
        return self._load_still_channel(still_source), None

    def _get_alignment_still(
        self, experiment: Experiment, still: np.ndarray, still_source: Path
    ) -> tuple[np.ndarray, Path]:
        """Return (alignment_array, alignment_source_path); falls back to still if not configured."""
        if not (self.auto_align and self.alignment_channel is not None):
            return still, still_source
        alignment_source = experiment.still_oir or still_source
        if self.still_loader is not None:
            return self.still_loader(
                alignment_source, self.alignment_channel, self.channel_axis, self.oir_axis_indices
            ), alignment_source
        if alignment_source.suffix.lower() == ".oir":
            # Oirfile remains isolated in the conversion environment. The
            # alignment channel is loaded into memory and its TIFF/sidecar are
            # removed with the temporary directory immediately afterward.
            with tempfile.TemporaryDirectory(prefix="suite2p_oir_alignment_") as tmp:
                tiff_path = convert_oir_in_environment(
                    alignment_source,
                    output=Path(tmp) / f"{experiment.name}_alignment.tif",
                    channel=self.alignment_channel,
                    axis_indices=self.oir_axis_indices,
                    environment=self.oir_environment,
                    converter_script=self.oir_converter_script,
                )
                return StillProcessor(path=tiff_path, channel=0).load(), alignment_source
        if self.alignment_channel == self.still_channel:
            return still, alignment_source
        return StillProcessor(
            path=alignment_source, channel=self.alignment_channel, channel_axis=self.channel_axis
        ).load(), alignment_source

    def _save_alignment_record(
        self,
        experiment: Experiment,
        alignment: dict[str, Any],
        selected_mode: str,
        dy: int,
        dx: int,
        alignment_source: Path,
    ) -> None:
        """Write still_alignment.json beside the plane outputs."""
        record = {
            **alignment,
            "segmentation_channel": self.still_channel,
            "segmentation_model": str(self.model_name_or_path),
            "cellprob_threshold": float(self.cellprob_threshold),
            "flow_threshold": float(self.flow_threshold),
            "alignment_channel": (
                self.still_channel if self.alignment_channel is None else self.alignment_channel
            ),
            "alignment_source": str(alignment_source),
            "requested_alignment_mode": self.alignment_mode,
            "selected_alignment_mode": selected_mode,
            "applied_residual_dy": dy if selected_mode == "affine" else 0,
            "applied_residual_dx": dx if selected_mode == "affine" else 0,
            "applied_dy": dy if selected_mode == "translation" else None,
            "applied_dx": dx if selected_mode == "translation" else None,
        }
        (experiment.plane_path / "still_alignment.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )

    def _save_qc(
        self,
        experiment: Experiment,
        reference_image: np.ndarray,
        masks: np.ndarray,
        alignment: dict[str, Any] | None,
        selected_mode: str,
        dy: int,
        dx: int,
    ) -> None:
        """Save cellpose_mask_overlay.png; no-op when save_qc is False."""
        if not self.save_qc:
            return
        alignment_title = (
            f"affine; residual dy={dy}, dx={dx}"
            if alignment is not None and selected_mode == "affine"
            else f"dy={dy}, dx={dx}"
        )
        axis = StillProcessor.plot_mask_overlay(
            reference_image, masks,
            title=f"{experiment.name}: still masks ({alignment_title})",
        )
        axis.figure.tight_layout()
        axis.figure.savefig(experiment.plane_path / "cellpose_mask_overlay.png", dpi=180)
        plt.close(axis.figure)

    def process(self, experiment: Experiment) -> dict[str, Any]:
        """Register one video, segment its still, and extract predefined ROIs."""
        if self.alignment_mode not in {"translation", "affine", "auto"}:
            raise ValueError(
                f"Unknown alignment mode {self.alignment_mode!r}; "
                "expected translation, affine, or auto"
            )
        still_source = experiment.still_source
        if still_source is None:
            raise FileNotFoundError(f"No matching {experiment.name}_snap still")

        db, registration_settings = self._build_registration_config(experiment)
        existing_db_path = experiment.plane_path / "db.npy"
        binary_exists = False
        if existing_db_path.is_file():
            existing_db = np.load(existing_db_path, allow_pickle=True).item()
            binary_exists = Path(existing_db.get("reg_file", "")).is_file()
        # A fresh binary must be freshly registered even if stale offsets remain
        # from an earlier run whose binary was deleted.
        registration_settings["run"]["do_registration"] = 1 if binary_exists else 2
        experiment.output_root.mkdir(parents=True, exist_ok=True)
        run_s2p(db=db, settings=registration_settings)

        plane_db, extraction_settings, reference_image = Suite2pInterface(
            experiment.plane_path
        ).load_context(image_file="reg_outputs.npy", image_key="meanImg")

        still, converted_still = self._load_segmentation_still(still_source)
        alignment_still, alignment_source = self._get_alignment_still(experiment, still, still_source)

        dy, dx = self.dy, self.dx
        alignment = None
        selected_mode = self.alignment_mode
        if self.auto_align:
            alignment = StillProcessor.estimate_alignment(
                reference_image,
                alignment_still,
                mode=self.alignment_mode,
                min_response=self.alignment_min_response,
                min_ecc=self.alignment_min_ecc,
                max_abs_shift=self.alignment_max_abs_shift,
            )
            selected_mode = alignment.get("selected_mode", self.alignment_mode)
            if selected_mode == "translation":
                dy += int(alignment["dy"])
                dx += int(alignment["dx"])

        masks_hires = StillProcessor(
            path=still_source,
            channel=self.still_channel,
            channel_axis=self.channel_axis,
            model_name_or_path=self.model_name_or_path,
            diameter=self.diameter,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
        ).segment(still)
        masks = StillProcessor.resize_label_masks(masks_hires, reference_image.shape)
        if alignment is not None and selected_mode == "affine":
            masks = StillProcessor.warp_label_masks(
                masks, alignment["warp_matrix"],
                target_shape=reference_image.shape, inverse_map=True,
            )
        masks = StillProcessor.shift_label_masks(masks, dy=dy, dx=dx)
        stat = Suite2pInterface(experiment.plane_path).masks_to_stats(
            masks, extraction_settings, do_soma_crop=False
        )

        np.save(experiment.plane_path / "cellpose_masks_hires.npy", masks_hires)
        np.save(experiment.plane_path / "cellpose_masks.npy", masks)
        if alignment is not None:
            self._save_alignment_record(experiment, alignment, selected_mode, dy, dx, alignment_source)
        self._save_qc(experiment, reference_image, masks, alignment, selected_mode, dy, dx)

        Suite2pInterface(experiment.plane_path).extract(
            stat=stat,
            output_path=experiment.plane_path,
            db=plane_db,
            settings=extraction_settings,
            device=self.device,
        )

        if self.delete_binary_after:
            binary = Path(plane_db["reg_file"])
            if binary.is_file():
                binary.unlink()
        return {
            "name": experiment.name,
            "folder": str(experiment.folder),
            "video": str(experiment.video),
            "still": str(still_source),
            "converted_still": str(converted_still) if converted_still else None,
            "output": str(experiment.plane_path),
            "status": "processed",
            "n_masks": int(masks.max()),
            "n_rois": int(len(stat)),
            "cellpose_model": str(self.model_name_or_path),
            "dy": dy,
            "dx": dx,
            "alignment_method": alignment["method"] if alignment else None,
            "selected_alignment_mode": selected_mode if alignment else None,
            "alignment_channel": self.alignment_channel,
            "alignment_response": alignment["response"] if alignment else None,
        }

    def run(self, root: str | Path) -> list[dict[str, Any]]:
        """Process all discovered videos beneath ``root``."""
        root = Path(root)
        experiments = discover_experiments(root, output_folder=self.output_folder)
        excluded = {name.casefold() for name in (self.exclude_names or ())}
        results: list[dict[str, Any]] = []
        oir_error = (
            oir_converter_runtime_error(self.oir_environment, self.oir_converter_script)
            if self.still_loader is None and any(
                experiment.name.casefold() not in excluded
                and experiment.still_tiff is None
                and experiment.still_oir is not None
                for experiment in experiments
            )
            else None
        )

        for experiment in experiments:
            base = {
                "name": experiment.name,
                "folder": str(experiment.folder),
                "video": str(experiment.video),
                "still_tiff": str(experiment.still_tiff) if experiment.still_tiff else None,
                "still_oir": str(experiment.still_oir) if experiment.still_oir else None,
                "output": str(experiment.plane_path),
            }
            if experiment.name.casefold() in excluded:
                results.append({**base, "status": "skipped_excluded"})
                continue
            if self.skip_completed and self.is_completed(experiment):
                results.append({**base, "status": "skipped_completed"})
                continue
            if experiment.still_source is None:
                results.append({**base, "status": "skipped_missing_still"})
                continue
            if experiment.still_tiff is None and oir_error is not None:
                results.append({**base, "status": "blocked_oirfile_runtime", "error": oir_error})
                continue
            if self.dry_run:
                results.append({**base, "status": "ready"})
                continue

            try:
                results.append(self.process(experiment))
            except Exception as exc:
                results.append({**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                if self.stop_on_error:
                    raise

        report_path = root / "still_cellpose_batch_results.json"
        report_path.write_text(
            json.dumps(
                [{k: str(v) if isinstance(v, Path) else v for k, v in row.items()} for row in results],
                indent=2,
            ),
            encoding="utf-8",
        )
        return results


def default_oir_converter_script() -> Path:
    """Return the standalone Oirfile conversion command path."""
    return Path(__file__).resolve().parents[1] / "scripts" / "oirfile_to_tiff.py"


def _conda_executable() -> str:
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not executable:
        raise FileNotFoundError("Could not locate the Conda executable")
    return executable


def oir_converter_runtime_error(
    environment: str = "image_conversion",
    converter_script: str | Path | None = None,
) -> str | None:
    """Return an error if the isolated Oirfile converter cannot start."""
    script = Path(converter_script or default_oir_converter_script())
    if not script.is_file():
        return f"Converter script not found: {script}"
    try:
        command = [
            _conda_executable(), "run", "-n", environment,
            "python", str(script), "--help",
        ]
        result = subprocess.run(
            command, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        return f"Converter environment failed ({result.returncode}): {detail}"
    return None


def convert_oir_in_environment(
    source: str | Path,
    *,
    output: str | Path | None = None,
    channel: int = 1,
    axis_indices: Mapping[str, int] | None = None,
    environment: str = "image_conversion",
    converter_script: str | Path | None = None,
) -> Path:
    """Invoke the Oirfile conversion stage and return its cached TIFF path."""
    source = Path(source)
    output = Path(output) if output is not None else source.with_suffix(".tif")
    script = Path(converter_script or default_oir_converter_script())
    command = [
        _conda_executable(), "run", "-n", environment,
        "python", str(script),
        "--input", str(source),
        "--output", str(output),
        "--channel", str(int(channel)),
    ]
    for axis, index in sorted((axis_indices or {}).items()):
        command.extend(["--axis-index", f"{str(axis).upper()}={int(index)}"])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"OIR conversion failed for {source} ({result.returncode}): {detail}"
        )
    if not output.is_file():
        raise FileNotFoundError(f"OIR converter did not create {output}")
    return output
