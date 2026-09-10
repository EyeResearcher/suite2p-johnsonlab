import numpy as np

from suite2p.still_cellpose import StillProcessor, Suite2pInterface


def test_predefined_extraction_enables_pipeline_stage(monkeypatch, tmp_path):
    import suite2p.still_cellpose as module

    reg_file = tmp_path / "data.bin"
    reg_file.touch()
    stat = np.array([{"ypix": np.array([0]), "xpix": np.array([0])}], dtype=object)
    captured = {}

    class FakeBinaryFile:
        def __init__(self, **kwargs):
            captured["binary_kwargs"] = kwargs

        def __enter__(self):
            return object()

        def __exit__(self, *args):
            return False

    def fake_pipeline(**kwargs):
        captured["settings"] = kwargs["settings"]
        captured["stat"] = kwargs["stat"]
        return (None,) * 11 + ({"total_plane_runtime": 0.0},)

    monkeypatch.setattr(module.io, "BinaryFile", FakeBinaryFile)
    monkeypatch.setattr(module, "pipeline", fake_pipeline)
    settings = {
        "run": {"do_detection": False},
        "extraction": {"lam_percentile": 50},
        "torch_device": "cpu",
    }
    db = {"reg_file": str(reg_file), "Ly": 2, "Lx": 3, "nframes": 4}

    Suite2pInterface(plane_path=tmp_path).extract(
        stat=stat,
        output_path=tmp_path / "output",
        db=db,
        settings=settings,
        device="cpu",
    )

    assert captured["settings"]["run"]["do_detection"] is True
    assert captured["settings"]["extraction"]["lam_percentile"] == 0
    assert captured["stat"] is stat


def test_estimate_still_shift_returns_inverse_translation():
    import cv2

    reference = np.zeros((128, 128), dtype=np.float32)
    for y, x, value in ((20, 30, 1.0), (67, 92, 0.7), (101, 45, 0.5)):
        reference[y, x] = value
    reference = cv2.GaussianBlur(reference, (0, 0), 2.0)
    transform = np.float32([[1, 0, -5], [0, 1, 3]])
    shifted_still = cv2.warpAffine(reference, transform, (128, 128))

    alignment = StillProcessor.estimate_shift(reference, shifted_still)

    assert alignment["dy"] == -3
    assert alignment["dx"] == 5
    assert alignment["response"] > 0.5


def test_estimate_still_affine_recovers_anisotropic_transform():
    import cv2

    rng = np.random.default_rng(7)
    reference = cv2.GaussianBlur(
        rng.normal(size=(256, 256)).astype(np.float32), (0, 0), 1.2
    )
    known = np.array(
        [[0.992, -0.001, 5.5], [0.001, 1.002, -2.0]], dtype=np.float32
    )
    still = cv2.warpAffine(
        reference,
        known,
        (256, 256),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    alignment = StillProcessor.estimate_affine(reference, still)
    recovered = np.asarray(alignment["warp_matrix"])

    assert alignment["ecc"] > 0.9
    np.testing.assert_allclose(recovered, known, atol=0.03)


def test_warp_labels_uses_ecc_inverse_map_convention():
    masks = np.zeros((64, 64), dtype=np.int32)
    masks[23:26, 35:38] = 7
    reference_to_still = np.array([[1, 0, 5], [0, 1, 3]], dtype=np.float32)

    aligned = StillProcessor.warp_label_masks(masks, reference_to_still, target_shape=(64, 64))

    assert np.all(aligned[20:23, 30:33] == 1)
    assert aligned.sum() == 9


def test_auto_alignment_keeps_translation_for_rigid_shift():
    import cv2

    rng = np.random.default_rng(11)
    reference = cv2.GaussianBlur(
        rng.normal(size=(256, 256)).astype(np.float32), (0, 0), 1.2
    )
    still = cv2.warpAffine(
        reference,
        np.array([[1, 0, -5], [0, 1, 3]], dtype=np.float32),
        (256, 256),
    )

    alignment = StillProcessor.estimate_alignment(reference, still, mode="auto")

    assert alignment["selected_mode"] == "translation"
    assert alignment["dy"] == -3
    assert alignment["dx"] == 5


def test_auto_alignment_selects_affine_for_position_dependent_error():
    import cv2

    rng = np.random.default_rng(13)
    reference = cv2.GaussianBlur(
        rng.normal(size=(256, 256)).astype(np.float32), (0, 0), 1.2
    )
    still = cv2.warpAffine(
        reference,
        np.array([[0.98, -0.002, 6], [0.002, 1.005, -2]], dtype=np.float32),
        (256, 256),
    )

    alignment = StillProcessor.estimate_alignment(reference, still, mode="auto")

    assert alignment["selected_mode"] == "affine"
    assert alignment["affine_correlation_gain"] > 0.02
    assert alignment["affine_deformation_pixels"] > 1.0


def test_select_channel_first_axis():
    image = np.stack((np.zeros((8, 9)), np.ones((8, 9))))
    selected = StillProcessor.select_channel(image, channel=1)
    np.testing.assert_array_equal(selected, np.ones((8, 9)))


def test_select_channel_last_axis():
    image = np.stack((np.zeros((5, 6)), np.ones((5, 6))), axis=-1)
    selected = StillProcessor.select_channel(image, channel=1, channel_axis=-1)
    np.testing.assert_array_equal(selected, np.ones((5, 6)))


def test_load_still_channel_accepts_cached_2d_tiff(tmp_path):
    import tifffile

    image = np.arange(20, dtype=np.uint16).reshape(4, 5)
    path = tmp_path / "selected_channel.tif"
    tifffile.imwrite(path, image)
    np.testing.assert_array_equal(StillProcessor(path=path, channel=1).load(), image)


def test_resize_labels_uses_nearest_neighbor():
    masks = np.array([[0, 1, 1, 0], [0, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    resized = StillProcessor.resize_label_masks(masks, (2, 2))
    np.testing.assert_array_equal(resized, np.array([[0, 1], [0, 0]]))


def test_shift_labels_does_not_wrap():
    masks = np.zeros((4, 4), dtype=np.int32)
    masks[0, 2] = 4
    shifted = StillProcessor.shift_label_masks(masks, dy=1, dx=-1)
    expected = np.zeros_like(masks)
    expected[1, 1] = 1
    np.testing.assert_array_equal(shifted, expected)


def test_relabel_masks_removes_gaps():
    masks = np.array([[0, 8], [3, 8]])
    np.testing.assert_array_equal(StillProcessor.relabel_masks(masks), np.array([[0, 2], [1, 2]]))


def test_relabel_masks_does_not_create_background_from_foreground():
    masks = np.array([[3, 8], [3, 8]])
    np.testing.assert_array_equal(StillProcessor.relabel_masks(masks), np.array([[1, 2], [1, 2]]))


def test_select_oir_plane_uses_named_c_axis():
    image = np.arange(2 * 4 * 5).reshape(2, 4, 5)
    selected = StillProcessor.select_oir_plane(image, dims=("C", "Y", "X"), channel=1)
    np.testing.assert_array_equal(selected, image[1])


def test_select_oir_plane_handles_singleton_tlz_axes():
    image = np.arange(1 * 1 * 1 * 2 * 4 * 5).reshape(1, 1, 1, 2, 4, 5)
    selected = StillProcessor.select_oir_plane(
        image, dims=("T", "L", "Z", "C", "Y", "X"), channel=1
    )
    np.testing.assert_array_equal(selected, image[0, 0, 0, 1])


def test_select_oir_plane_requires_index_for_non_singleton_z():
    image = np.zeros((3, 2, 4, 5))
    try:
        StillProcessor.select_oir_plane(image, dims=("Z", "C", "Y", "X"), channel=1)
    except ValueError as exc:
        assert "axis 'Z' has 3 planes" in str(exc)
    else:
        raise AssertionError("Expected an explicit Z-index error")


def test_select_oir_plane_applies_explicit_z_index():
    image = np.arange(3 * 2 * 4 * 5).reshape(3, 2, 4, 5)
    selected = StillProcessor.select_oir_plane(
        image,
        dims=("Z", "C", "Y", "X"),
        channel=1,
        axis_indices={"Z": 2},
    )
    np.testing.assert_array_equal(selected, image[2, 1])


def test_select_oir_plane_transposes_named_xy_to_yx():
    image = np.arange(5 * 4 * 2).reshape(5, 4, 2)
    selected = StillProcessor.select_oir_plane(image, dims=("X", "Y", "C"), channel=1)
    np.testing.assert_array_equal(selected, image[:, :, 1].T)
