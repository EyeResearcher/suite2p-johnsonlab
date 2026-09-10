"""Convert one named OIR plane to a 2-D YX TIFF using Oirfile.

This script is an environment boundary. It is intended to run in the
``image_conversion`` Conda environment, independently of Suite2p.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import oirfile
from oirfile import OirFile
import tifffile


def parse_axis_indices(values: Sequence[str]) -> dict[str, int]:
    """Parse repeated ``AXIS=INDEX`` arguments."""
    result: dict[str, int] = {}
    for value in values:
        try:
            axis, index = value.split("=", 1)
            axis = axis.strip().upper()
            if axis not in {"T", "L", "Z"}:
                raise ValueError
            result[axis] = int(index)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid axis index {value!r}; expected T=0, L=0, or Z=0"
            ) from exc
    return result


def select_named_plane(
    image: np.ndarray,
    dims: Sequence[str],
    channel: int,
    axis_indices: Mapping[str, int],
) -> np.ndarray:
    """Select channel and optional TLZ indices, returning explicit YX order."""
    image = np.asarray(image)
    dims = tuple(str(dim).upper() for dim in dims)
    if len(dims) != image.ndim:
        raise ValueError(f"Dimensions {dims} do not match shape {image.shape}")
    if len(set(dims)) != len(dims) or not {"Y", "X"}.issubset(dims):
        raise ValueError(f"Invalid OIR dimensions: {dims}")

    channel_dim = "C" if "C" in dims else "S" if "S" in dims else None
    selections: list[int | slice] = []
    remaining: list[str] = []
    for dim, size in zip(dims, image.shape):
        if dim in {"Y", "X"}:
            selections.append(slice(None))
            remaining.append(dim)
            continue
        if dim == channel_dim:
            index = channel
        elif size == 1:
            index = 0
        elif dim in axis_indices:
            index = int(axis_indices[dim])
        else:
            raise ValueError(
                f"OIR axis {dim!r} has {size} planes; pass --axis-index {dim}=<index>"
            )
        if not 0 <= index < size:
            raise IndexError(
                f"Index {index} is unavailable for axis {dim!r} with size {size}"
            )
        selections.append(index)

    if channel_dim is None and channel != 0:
        raise IndexError(
            f"Requested channel {channel}, but dimensions contain no C/S axis: {dims}"
        )
    selected = image[tuple(selections)]
    if tuple(remaining) != ("Y", "X"):
        selected = np.transpose(
            selected, (remaining.index("Y"), remaining.index("X"))
        )
    if selected.ndim != 2:
        raise ValueError(f"Selection is not 2-D YX: {selected.shape}")
    return selected


def open_oir(path: Path) -> OirFile:
    """Open OIR using arguments supported by the installed Oirfile version."""
    parameters = inspect.signature(OirFile).parameters
    kwargs = {"squeeze": False}
    if "multifile" in parameters:
        kwargs["multifile"] = True
    return OirFile(path, **kwargs)


def sidecar_path(output: Path) -> Path:
    return output.with_suffix(".conversion.json")


def source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_matches(
    source: Path,
    output: Path,
    channel: int,
    axis_indices: Mapping[str, int],
) -> bool:
    metadata_path = sidecar_path(output)
    if not output.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        metadata.get("source_signature") == source_signature(source)
        and metadata.get("channel_index") == channel
        and metadata.get("axis_indices") == dict(axis_indices)
        and metadata.get("output_dims") == ["Y", "X"]
    )


def convert(
    source: Path,
    output: Path,
    channel: int,
    axis_indices: Mapping[str, int],
    overwrite: bool = False,
) -> dict:
    """Convert one OIR selection and return conversion metadata."""
    source, output = source.resolve(), output.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not overwrite and cache_matches(source, output, channel, axis_indices):
        return json.loads(sidecar_path(output).read_text(encoding="utf-8"))

    with open_oir(source) as oir:
        dims = tuple(oir.dims)
        shape = tuple(int(size) for size in oir.shape)
        image = oir.asarray()
        channel_names = [
            str(getattr(item, "name", "")) for item in getattr(oir, "channels", ())
        ]
    selected = select_named_plane(image, dims, channel, axis_indices)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.tif")
    tifffile.imwrite(temporary, selected, metadata={"axes": "YX"})
    os.replace(temporary, output)

    metadata = {
        "source": str(source),
        "source_signature": source_signature(source),
        "oirfile_version": oirfile.__version__,
        "original_dims": list(dims),
        "original_shape": list(shape),
        "channel_names": channel_names,
        "channel_axis": "C" if "C" in dims else "S" if "S" in dims else None,
        "channel_index": channel,
        "axis_indices": dict(axis_indices),
        "output": str(output),
        "output_dims": ["Y", "X"],
        "output_shape": list(selected.shape),
        "dtype": str(selected.dtype),
    }
    sidecar_path(output).write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one OIR channel/plane to a 2-D YX TIFF"
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument(
        "--axis-index",
        action="append",
        default=[],
        metavar="AXIS=INDEX",
        help="Select a non-singleton T, L, or Z plane; may be repeated",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    axis_indices = parse_axis_indices(args.axis_index)
    metadata = convert(
        args.input,
        args.output,
        channel=args.channel,
        axis_indices=axis_indices,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
