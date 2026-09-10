"""Use Cellpose masks from a high-resolution still for Suite2p extraction."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import io
from .pipeline_s2p import pipeline
from .detection.anatomical import masks_to_stats as _masks_to_stats
from .detection.stats import roi_stats



@dataclass
class _Validate:
    """Centralised input-checking helpers — each method raises on failure and returns the array."""

    @staticmethod
    def mask_2d(arr: np.ndarray, name: str = "label image") -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D {name}, got {arr.shape}")
        return arr

    @staticmethod
    def mask_2d_nonneg(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D label image, got {arr.shape}")
        if np.any(arr < 0):
            raise ValueError("Mask labels must be nonnegative")
        return arr

    @staticmethod
    def float32_image_2d(arr: np.ndarray, name: str = "image") -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32).squeeze()
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2-D {name}, got {arr.shape}")
        return arr

    @staticmethod
    def target_shape_2d(shape: Sequence[int]) -> tuple[int, int]:
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError(f"Invalid target shape: {shape}")
        return int(shape[0]), int(shape[1])

    @staticmethod
    def affine_2x3(arr: Any, *, dtype: type = np.float32) -> np.ndarray:
        arr = np.asarray(arr, dtype=dtype)
        if arr.shape != (2, 3):
            raise ValueError(f"Expected a 2x3 affine matrix, got {arr.shape}")
        return arr

    @staticmethod
    def all_finite(values: tuple, label: str) -> None:
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"Non-finite {label}: {values}")

    @staticmethod
    def oir_index_in_bounds(dim: str, index: int, size: int) -> None:
        if not 0 <= index < size:
            raise IndexError(f"Index {index} is unavailable for OIR axis {dim!r} with size {size}")


@dataclass
class StillProcessor:
    """Load a microscopy still, run Cellpose, and produce aligned video-resolution masks."""

    path: Path | str
    channel: int = 1
    channel_axis: int | None = None
    model_name_or_path: str = "cpsam"
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @staticmethod
    def relabel_masks(masks: np.ndarray) -> np.ndarray:
        """Replace arbitrary nonnegative labels with consecutive integer labels."""
        masks = _Validate.mask_2d_nonneg(masks)
        labels = np.unique(masks)
        labels = labels[labels > 0]
        relabeled = np.zeros(masks.shape, dtype=np.int32)
        foreground = masks > 0
        relabeled[foreground] = np.searchsorted(labels, masks[foreground]) + 1
        return relabeled

    @staticmethod
    def select_channel(
        image: np.ndarray,
        channel: int = 1,
        channel_axis: int | None = None,
        max_auto_channels: int = 4,
    ) -> np.ndarray:
        """Select a zero-based channel from a 3-D multichannel image."""
        image = np.squeeze(np.asarray(image))
        if image.ndim == 2:
            raise ValueError("Image is 2-D; no channel can be selected")
        if image.ndim != 3:
            raise ValueError(f"Expected a 3-D multichannel image, got {image.shape}")
        if channel_axis is None:
            candidates = [
                axis for axis, size in enumerate(image.shape)
                if size <= max_auto_channels
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Cannot identify the channel axis from {image.shape}; "
                    "set channel_axis explicitly"
                )
            channel_axis = candidates[0]
        channel_axis %= image.ndim
        if not 0 <= channel < image.shape[channel_axis]:
            raise IndexError(
                f"Channel {channel} is unavailable on axis {channel_axis} "
                f"of image shape {image.shape}"
            )
        selected = np.take(image, channel, axis=channel_axis)
        if selected.ndim != 2:
            raise ValueError(f"Selected channel is not 2-D: {selected.shape}")
        return selected

    @staticmethod
    def select_oir_plane(
        image: np.ndarray,
        dims: Sequence[str],
        channel: int = 1,
        axis_indices: Mapping[str, int] | None = None,
    ) -> np.ndarray:
        """Select a 2-D ``YX`` plane using Oirfile's named dimensions.

        Oirfile reports dimensions from ``T, L, Z, C/S, Y, X`` while omitting
        dimensions not present in the acquisition. Non-spatial singleton axes are
        selected automatically. Non-singleton T, L, or Z axes require an explicit
        entry in ``axis_indices`` to avoid silently selecting the wrong plane.
        """
        image = np.asarray(image)
        dims = tuple(str(dim).upper() for dim in dims)
        axis_indices = {
            str(dim).upper(): int(index)
            for dim, index in (axis_indices or {}).items()
        }
        if len(dims) != image.ndim:
            raise ValueError(
                f"OIR dimensions {dims} do not match image shape {image.shape}"
            )
        if len(set(dims)) != len(dims):
            raise ValueError(f"OIR dimensions must be unique: {dims}")
        if "Y" not in dims or "X" not in dims:
            raise ValueError(f"OIR image must contain Y and X dimensions: {dims}")
        channel_dim = "C" if "C" in dims else "S" if "S" in dims else None
        selections: list[int | slice] = []
        remaining_dims: list[str] = []
        for axis, (dim, size) in enumerate(zip(dims, image.shape)):
            if dim in {"Y", "X"}:
                selections.append(slice(None))
                remaining_dims.append(dim)
                continue
            if dim == channel_dim:
                index = int(channel)
            elif size == 1:
                index = 0
            elif dim in axis_indices:
                index = axis_indices[dim]
            else:
                raise ValueError(
                    f"OIR axis {dim!r} has {size} planes. Set "
                    f"axis_indices={{'{dim}': <index>}} explicitly."
                )
            _Validate.oir_index_in_bounds(dim, index, size)
            selections.append(index)
        if channel_dim is None and channel != 0:
            raise IndexError(
                f"Requested channel {channel}, but OIR dimensions contain no C/S axis: {dims}"
            )
        selected = image[tuple(selections)]
        if tuple(remaining_dims) != ("Y", "X"):
            selected = np.transpose(
                selected,
                (remaining_dims.index("Y"), remaining_dims.index("X")),
            )
        if selected.ndim != 2:
            raise ValueError(
                f"OIR selection did not produce a 2-D YX image: {selected.shape}"
            )
        return selected

    @staticmethod
    def resize_label_masks(masks: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
        """Resize integer labels exactly with nearest-neighbor index mapping."""
        masks = _Validate.mask_2d(masks)
        target_y, target_x = _Validate.target_shape_2d(target_shape)
        src_y, src_x = masks.shape
        yi = np.minimum((np.arange(target_y) * src_y / target_y).astype(int), src_y - 1)
        xi = np.minimum((np.arange(target_x) * src_x / target_x).astype(int), src_x - 1)
        return StillProcessor.relabel_masks(masks[np.ix_(yi, xi)])

    @staticmethod
    def shift_label_masks(masks: np.ndarray, dy: int = 0, dx: int = 0) -> np.ndarray:
        """Translate labels without wraparound; positive values move down/right."""
        masks = _Validate.mask_2d(masks)
        dy, dx = int(dy), int(dx)
        h, w = masks.shape
        shifted = np.zeros_like(masks)
        sy0, sy1 = max(0, -dy), min(h, h - dy)
        sx0, sx1 = max(0, -dx), min(w, w - dx)
        if sy1 > sy0 and sx1 > sx0:
            shifted[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = masks[sy0:sy1, sx0:sx1]
        return StillProcessor.relabel_masks(shifted)

    @staticmethod
    def warp_label_masks(
        masks: np.ndarray,
        warp_matrix: Sequence[Sequence[float]],
        target_shape: Sequence[int] | None = None,
        *,
        inverse_map: bool = True,
    ) -> np.ndarray:
        """Apply an affine transform to integer labels without interpolating IDs."""
        import cv2

        masks = _Validate.mask_2d(masks)
        matrix = _Validate.affine_2x3(warp_matrix)
        if target_shape is None:
            target_shape = masks.shape
        target_y, target_x = map(int, target_shape)
        flags = cv2.INTER_NEAREST | (cv2.WARP_INVERSE_MAP if inverse_map else 0)
        warped = cv2.warpAffine(
            masks.astype(np.float32),
            matrix,
            (target_x, target_y),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return StillProcessor.relabel_masks(np.rint(warped).astype(np.int64))

    @staticmethod
    def _prepare_alignment_image(image: np.ndarray) -> np.ndarray:
        """Normalize and high-pass an image for multimodal registration."""
        import cv2
        image = np.asarray(image, dtype=np.float32)
        low, high = np.percentile(image, (1.0, 99.8))
        logged = np.log1p(20.0 * np.clip((image - low) / (high - low + 1e-6), 0, 1)).astype(np.float32)
        return np.asarray(logged - cv2.GaussianBlur(logged, (0, 0), sigmaX=5.0, sigmaY=5.0), dtype=np.float32)

    @staticmethod
    def _resize_for_alignment(
        reference_image: np.ndarray, still_image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return float32 2-D images with the still resized to the reference shape."""
        import cv2
        reference = _Validate.float32_image_2d(reference_image, "reference image")
        still = _Validate.float32_image_2d(still_image, "still image")
        return reference, cv2.resize(still, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _alignment_correlation(
        reference_image: np.ndarray,
        aligned_image: np.ndarray,
        *,
        border: int = 24,
    ) -> float:
        """Return Pearson correlation of prepared images away from warped edges."""
        ref = StillProcessor._prepare_alignment_image(reference_image)
        aln = StillProcessor._prepare_alignment_image(aligned_image)
        if ref.shape != aln.shape:
            raise ValueError(f"Aligned image shape {aln.shape} does not match {ref.shape}")
        border = max(0, int(border))
        if border and min(ref.shape) > 2 * border:
            ref, aln = ref[border:-border, border:-border], aln[border:-border, border:-border]
        ref, aln = ref.ravel(), aln.ravel()
        if ref.std() <= 1e-12 or aln.std() <= 1e-12:
            raise ValueError("Cannot score alignment of a constant image")
        correlation = float(np.corrcoef(ref, aln)[0, 1])
        if not np.isfinite(correlation):
            raise ValueError(f"Non-finite alignment correlation: {correlation}")
        return correlation

    @staticmethod
    def _affine_deformation_pixels(
        warp_matrix: Sequence[Sequence[float]], target_shape: Sequence[int]
    ) -> float:
        """Measure position-dependent affine displacement after removing translation."""
        matrix = _Validate.affine_2x3(warp_matrix, dtype=np.float64)
        ty, tx = map(int, target_shape)
        corners = np.array(
            [[-0.5*tx, -0.5*ty], [0.5*tx, -0.5*ty], [-0.5*tx, 0.5*ty], [0.5*tx, 0.5*ty]],
            dtype=np.float64,
        )
        return float(np.linalg.norm(corners @ (matrix[:, :2] - np.eye(2)).T, axis=1).max())

    @staticmethod
    def estimate_shift(
        reference_image: np.ndarray,
        still_image: np.ndarray,
        *,
        min_response: float = 0.1,
        max_abs_shift: float = 50.0,
    ) -> dict[str, float | int | str]:
        """Estimate the integer translation that aligns a still to a video image.

        Returns ``dy`` and ``dx`` as the inverse correction to apply to masks.
        """
        import cv2
        reference, resized = StillProcessor._resize_for_alignment(reference_image, still_image)
        (rdx, rdy), response = cv2.phaseCorrelate(
            StillProcessor._prepare_alignment_image(reference),
            StillProcessor._prepare_alignment_image(resized),
        )
        _Validate.all_finite((rdy, rdx, response), "still alignment result")
        if response < min_response:
            raise ValueError(f"Still alignment response {response:.3f} is below {min_response:.3f}")
        if max(abs(rdy), abs(rdx)) > max_abs_shift:
            raise ValueError(f"Still displacement (dy={rdy:.2f}, dx={rdx:.2f}) exceeds {max_abs_shift:.1f} pixels")
        return {
            "method": "phase_correlation_log_highpass",
            "relative_dy": float(rdy), "relative_dx": float(rdx),
            "response": float(response),
            "dy": int(np.rint(-rdy)), "dx": int(np.rint(-rdx)),
        }

    @staticmethod
    def estimate_affine(
        reference_image: np.ndarray,
        still_image: np.ndarray,
        *,
        min_ecc: float = 0.5,
        max_abs_translation: float = 50.0,
        max_scale_deviation: float = 0.05,
        max_abs_rotation: float = 2.0,
        max_abs_shear: float = 0.02,
        max_iterations: int = 300,
        epsilon: float = 1e-7,
    ) -> dict[str, Any]:
        """Estimate a validated affine transform from a still to a video image.

        The returned matrix follows OpenCV ECC convention: it maps reference
        coordinates to resized-still coordinates and must therefore be applied to
        the still or its masks with ``cv2.WARP_INVERSE_MAP``.
        """
        import cv2
        reference, resized = StillProcessor._resize_for_alignment(reference_image, still_image)
        # Phase correlation initializes translation; ECC refines to full affine.
        initial = StillProcessor.estimate_shift(reference, resized, min_response=0.0, max_abs_shift=max_abs_translation)
        warp = np.array(
            [[1.0, 0.0, float(initial["relative_dx"])], [0.0, 1.0, float(initial["relative_dy"])]],
            dtype=np.float32,
        )
        try:
            ecc, warp = cv2.findTransformECC(
                StillProcessor._prepare_alignment_image(reference),
                StillProcessor._prepare_alignment_image(resized),
                warp, cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(max_iterations), float(epsilon)),
                None, 5,
            )
        except cv2.error as exc:
            raise ValueError(f"Affine still alignment failed: {exc}") from exc

        linear = np.asarray(warp[:, :2], dtype=np.float64)
        scale_x = float(np.linalg.norm(linear[:, 0]))
        scale_y = float(np.linalg.norm(linear[:, 1]))
        rotation_deg = float(np.degrees(np.arctan2(linear[1, 0], linear[0, 0])))
        shear_cos = float(np.dot(linear[:, 0], linear[:, 1]) / (scale_x * scale_y + 1e-12))
        tx, ty = map(float, warp[:, 2])
        _Validate.all_finite((ecc, scale_x, scale_y, rotation_deg, shear_cos, tx, ty), "affine alignment result")
        for condition, message in [
            (ecc < min_ecc,
             f"Affine alignment ECC {ecc:.3f} is below {min_ecc:.3f}"),
            (max(abs(tx), abs(ty)) > max_abs_translation,
             f"Affine alignment translation (y={ty:.2f}, x={tx:.2f}) exceeds {max_abs_translation:.1f} pixels"),
            (max(abs(scale_x - 1.0), abs(scale_y - 1.0)) > max_scale_deviation,
             f"Affine alignment scale ({scale_y:.5f}, {scale_x:.5f}) exceeds {max_scale_deviation:.3f}"),
            (abs(rotation_deg) > max_abs_rotation,
             f"Affine alignment rotation {rotation_deg:.3f} deg exceeds {max_abs_rotation:.3f} deg"),
            (abs(shear_cos) > max_abs_shear,
             f"Affine alignment shear {shear_cos:.5f} exceeds {max_abs_shear:.5f}"),
        ]:
            if condition:
                raise ValueError(message)
        return {
            "method": "ecc_affine_log_highpass",
            "response": float(ecc), "ecc": float(ecc),
            "warp_matrix": np.asarray(warp, dtype=float).tolist(),
            "warp_direction": "reference_to_resized_still_use_inverse_map",
            "scale_x": scale_x, "scale_y": scale_y,
            "rotation_deg": rotation_deg, "shear_cos": shear_cos,
            "translation_x": tx, "translation_y": ty,
            "initial_phase_alignment": initial,
        }

    @staticmethod
    def estimate_alignment(
        reference_image: np.ndarray,
        still_image: np.ndarray,
        *,
        mode: str = "translation",
        min_response: float = 0.1,
        min_ecc: float = 0.5,
        max_abs_shift: float = 50.0,
        min_affine_correlation_gain: float = 0.02,
        min_affine_deformation: float = 1.0,
    ) -> dict[str, Any]:
        """Estimate alignment; mode is 'translation', 'affine', or 'auto'."""
        import cv2
        if mode == "affine":
            return StillProcessor.estimate_affine(
                reference_image, still_image, min_ecc=min_ecc, max_abs_translation=max_abs_shift,
            )
        if mode == "translation":
            return StillProcessor.estimate_shift(
                reference_image, still_image, min_response=min_response, max_abs_shift=max_abs_shift,
            )
        # auto: use translation unless affine provides meaningful correlation gain and deformation
        reference, resized = StillProcessor._resize_for_alignment(reference_image, still_image)
        translation = StillProcessor.estimate_shift(reference, resized, min_response=min_response, max_abs_shift=max_abs_shift)
        translated = cv2.warpAffine(
            resized,
            np.array([[1.0, 0.0, float(translation["dx"])], [0.0, 1.0, float(translation["dy"])]], dtype=np.float32),
            (reference.shape[1], reference.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        translation_correlation = StillProcessor._alignment_correlation(reference, translated)
        affine = affine_correlation = correlation_gain = deformation = affine_error = None
        use_affine = False
        try:
            affine = StillProcessor.estimate_affine(reference, resized, min_ecc=min_ecc, max_abs_translation=max_abs_shift)
            affine_warp = np.asarray(affine["warp_matrix"], dtype=np.float32)
            affine_aligned = cv2.warpAffine(
                resized, affine_warp, (reference.shape[1], reference.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
            )
            affine_correlation = StillProcessor._alignment_correlation(reference, affine_aligned)
            correlation_gain = affine_correlation - translation_correlation
            deformation = StillProcessor._affine_deformation_pixels(affine["warp_matrix"], reference.shape)
            use_affine = correlation_gain >= min_affine_correlation_gain and deformation >= min_affine_deformation
        except ValueError as exc:
            affine_error = f"{type(exc).__name__}: {exc}"
        selected = dict(affine if use_affine else translation)
        selected.update({
            "selection_method": "translation_or_affine_correlation",
            "selected_mode": "affine" if use_affine else "translation",
            "translation_correlation": translation_correlation,
            "affine_correlation": affine_correlation,
            "affine_correlation_gain": correlation_gain,
            "affine_deformation_pixels": deformation,
            "affine_candidate_error": affine_error,
            "min_affine_correlation_gain": min_affine_correlation_gain,
            "min_affine_deformation": min_affine_deformation,
        })
        return selected

    @staticmethod
    def resolve_model(
        model_name_or_path: str | Path,
        *,
        hf_repo_id: str | None = None,
        hf_subfolder: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> str:
        """Return a local Cellpose model path, fetching from HuggingFace when needed.

        If ``model_name_or_path`` is an existing local file it is returned unchanged.
        If ``hf_repo_id`` is provided and the path does not exist locally, the file
        is downloaded via ``huggingface_hub`` (respects ``HF_TOKEN`` for private repos).
        Otherwise the value is returned as-is so Cellpose can resolve built-in names
        such as ``'cpsam'`` or ``'cyto3'`` through its own download mechanism.
        """
        path = Path(model_name_or_path)
        if path.is_file():
            return str(path)
        if hf_repo_id is not None:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise ImportError(
                    "Downloading models from HuggingFace requires the 'huggingface_hub' package. "
                    "Install it with: pip install huggingface_hub"
                ) from exc
            local_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=str(model_name_or_path),
                subfolder=hf_subfolder,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
            )
            return local_path
        return str(model_name_or_path)

    def load(self) -> np.ndarray:
        """Return the 2-D still channel from disk."""
        path = self.path
        if not path.is_file():
            raise FileNotFoundError(path)
        image = np.squeeze(np.asarray(iio.imread(path)))
        if image.ndim == 2:
            return image
        return self.select_channel(image, channel=self.channel, channel_axis=self.channel_axis)

    def load_oir(
        self,
        axis_indices: Mapping[str, int] | None = None,
        convert_path: str | Path | None = None,
        multifile: bool = True,
        memmap: bool = False,
    ) -> np.ndarray:
        """Read the OIR file at self.path and optionally cache it as TIFF."""
        try:
            from oirfile import OirFile
        except (ImportError, SyntaxError) as exc:
            raise ImportError(
                "Reading .oir snapshots requires the 'oirfile' package. "
                "Install it in a Python 3.12+ environment."
            ) from exc
        import tifffile

        path = self.path
        if not path.is_file():
            raise FileNotFoundError(path)
        with OirFile(path, squeeze=False, multifile=multifile, memmap=memmap) as oir:
            dims = tuple(oir.dims)
            image = oir.asarray()
        selected = self.select_oir_plane(
            image, dims=dims, channel=self.channel, axis_indices=axis_indices
        )
        if convert_path is not None:
            convert_path = Path(convert_path)
            convert_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(convert_path, selected, metadata={"axes": "YX"})
        return selected

    def load_cellpose_masks(self, mask_path: str | Path) -> np.ndarray:
        """Load either a plain label array or a Cellpose ``*_seg.npy`` dict."""
        loaded = np.load(mask_path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            loaded = loaded.item()
        masks = loaded["masks"] if isinstance(loaded, Mapping) else loaded
        masks = np.asarray(masks, dtype=np.int32).squeeze()
        masks = _Validate.mask_2d(masks, "Cellpose label image")
        return self.relabel_masks(masks)

    def segment(self, still: np.ndarray, gpu: bool | None = None, **eval_kwargs: Any) -> np.ndarray:
        """Run Cellpose on a 2-D still and return hires label masks."""
        from cellpose import core
        from cellpose.models import CellposeModel

        still = _Validate.mask_2d(still, "still")
        use_gpu = bool(core.use_gpu()) if gpu is None else gpu
        model = CellposeModel(pretrained_model=self.model_name_or_path, gpu=use_gpu)
        masks = model.eval(
            still,
            diameter=self.diameter,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            **eval_kwargs,
        )[0]
        return self.relabel_masks(masks)

    def align_masks(
        self,
        masks: np.ndarray,
        target_shape: Sequence[int],
        dy: int = 0,
        dx: int = 0,
        alignment: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Resize hires masks and apply translation or affine alignment correction."""
        masks = self.resize_label_masks(masks, target_shape)
        if alignment is not None:
            # affine mode uses warp; translation mode accumulates into dy/dx
            if alignment.get("selected_mode", "translation") == "affine":
                masks = self.warp_label_masks(
                    masks, alignment["warp_matrix"],
                    target_shape=target_shape, inverse_map=True,
                )
            else:
                dy += int(alignment["dy"])
                dx += int(alignment["dx"])
        return self.shift_label_masks(masks, dy=dy, dx=dx)

    @staticmethod
    def plot_mask_overlay(
        image: np.ndarray,
        masks: np.ndarray,
        title: str | None = None,
        ax: Any = None,
        color: str = "red",
        linewidth: float = 0.5,
    ) -> Any:
        """Plot label boundaries over an image and return the Matplotlib axis."""
        image, masks = np.asarray(image), np.asarray(masks)
        if image.shape != masks.shape:
            raise ValueError(f"Image {image.shape} and masks {masks.shape} differ")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap="gray")
        ax.contour(masks > 0, levels=[0.5], colors=color, linewidths=linewidth)
        ax.set_title(title or "Mask overlay")
        ax.axis("off")
        return ax


@dataclass
class Suite2pInterface:
    """Load Suite2p plane context, convert masks to stats, and run extraction."""

    plane_path: Path | str

    def __post_init__(self) -> None:
        self.plane_path = Path(self.plane_path)

    def load_context(
        self,
        image_file: str = "reg_outputs.npy",
        image_key: str = "meanImg",
    ) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
        """Return (db, settings, reference_image) for this plane."""
        plane_path = self.plane_path
        required = ("db.npy", "settings.npy", image_file)
        missing = [name for name in required if not (plane_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing from {plane_path}: {', '.join(missing)}")
        db = np.load(plane_path / "db.npy", allow_pickle=True).item()
        settings = np.load(plane_path / "settings.npy", allow_pickle=True).item()
        image_outputs = np.load(plane_path / image_file, allow_pickle=True).item()
        if image_key not in image_outputs:
            raise KeyError(f"{image_key!r} is not present in {image_file}")
        reference_image = np.asarray(image_outputs[image_key]).squeeze()
        if reference_image.ndim != 2:
            raise ValueError(f"Expected a 2-D reference image, got {reference_image.shape}")
        return db, settings, reference_image

    def masks_to_stats(
        self,
        masks: np.ndarray,
        settings: Mapping[str, Any],
        do_soma_crop: bool = False,
    ) -> np.ndarray:
        """Convert a video-resolution label image to a Suite2p stat array."""
        masks = StillProcessor.relabel_masks(masks)
        if masks.max() == 0:
            raise ValueError("No ROI labels are present")
        height, width = masks.shape
        weights = np.ones((height, width), dtype=np.float32)
        stat = _masks_to_stats(masks, weights)
        detection_settings = settings["detection"]
        return roi_stats(
            stat,
            Ly=height,
            Lx=width,
            diameter=settings["diameter"],
            max_overlap=detection_settings.get("max_overlap", 0.75),
            do_soma_crop=do_soma_crop,
            npix_norm_min=detection_settings.get("npix_norm_min", 0.0),
            npix_norm_max=detection_settings.get("npix_norm_max", 100.0),
            median=True,
        )

    def extract(
        self,
        stat: np.ndarray,
        output_path: Path | str | None = None,
        db: Mapping[str, Any] | None = None,
        settings: Mapping[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> tuple[Any, ...]:
        """Extract Suite2p traces from registered data using predefined ROIs."""
        plane_path = self.plane_path
        output_path = Path(output_path) if output_path is not None else plane_path
        if db is None:
            db = np.load(plane_path / "db.npy", allow_pickle=True).item()
        if settings is None:
            settings = np.load(plane_path / "settings.npy", allow_pickle=True).item()
        db, settings = deepcopy(dict(db)), deepcopy(dict(settings))
        settings["extraction"]["lam_percentile"] = 0
        # Registration-only batch runs persist do_detection=False. Supplying
        # predefined stats already bypasses Suite2p's ROI detector, but the pipeline
        # stage must be enabled so extraction, deconvolution, classification, and
        # the conventional F/Fneu/spks/stat/iscell saves are reached. This matches
        # the full-run settings used by the original validation notebook.
        settings["run"]["do_detection"] = True

        reg_file = Path(db["reg_file"])
        if not reg_file.is_file():
            raise FileNotFoundError(f"Registered binary not found: {reg_file}")
        output_path.mkdir(parents=True, exist_ok=True)

        height, width = int(db["Ly"]), int(db["Lx"])
        if device is None:
            configured = settings.get("torch_device", "cuda")
            configured = configured if isinstance(configured, str) else "cuda"
            device = configured if configured != "cuda" or torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        with io.BinaryFile(
            Ly=height,
            Lx=width,
            filename=reg_file,
            n_frames=db["nframes"],
            write=False,
        ) as registered_movie:
            outputs = pipeline(
                save_path=str(output_path),
                f_reg=registered_movie,
                run_registration=False,
                settings=settings,
                stat=stat,
                device=device,
            )

        np.save(output_path / "db.npy", db)
        np.save(output_path / "settings.npy", settings)
        reg_outputs, detect_outputs = outputs[:2]
        plane_times = outputs[-1]
        ops = {
            **db,
            **settings,
            **(reg_outputs or {}),
            **(detect_outputs or {}),
            **plane_times,
        }
        np.save(output_path / "ops.npy", ops)
        return outputs

    def plot_corrected_traces(
        self,
        output_path: str | Path | None = None,
        settings: Mapping[str, Any] | None = None,
        n_traces: int = 10,
        indices: Sequence[int] | None = None,
        seed: int | None = 0,
        ax: Any = None,
    ) -> Any:
        """Plot neuropil-corrected traces and return the Matplotlib axis."""
        output_path = Path(output_path) if output_path is not None else self.plane_path
        if settings is None:
            settings = np.load(self.plane_path / "settings.npy", allow_pickle=True).item()
        fluorescence = np.load(output_path / "F.npy")
        neuropil = np.load(output_path / "Fneu.npy")
        coefficient = settings["extraction"].get("neuropil_coefficient", 0.7)
        corrected = fluorescence - coefficient * neuropil
        if len(corrected) == 0:
            raise ValueError("No traces were extracted")
        if indices is None:
            count = min(int(n_traces), len(corrected))
            indices = np.random.default_rng(seed).choice(
                len(corrected), size=count, replace=False
            )
        else:
            indices = np.asarray(indices, dtype=int)
        offset = np.nanpercentile(np.abs(corrected[indices]), 95)
        offset = float(offset) if np.isfinite(offset) and offset > 0 else 1.0
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 8))
        for row, index in enumerate(indices):
            ax.plot(corrected[index] + row * offset, linewidth=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Corrected fluorescence (offset)")
        ax.set_title("Still-mask Suite2p traces")
        return ax
