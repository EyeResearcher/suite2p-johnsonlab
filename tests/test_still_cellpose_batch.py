from pathlib import Path

from suite2p.still_cellpose_batch import (
    BatchProcessor,
    deep_merge,
    discover_experiments,
)


def test_discover_experiments_pairs_video_and_nested_tiff_still(tmp_path):
    folder = tmp_path / "BP" / "1-4_Day5"
    snap = folder / "snap"
    snap.mkdir(parents=True)
    (folder / "1-4_Day5.tiff").touch()
    (snap / "1-4_Day5_snap.tif").touch()
    (snap / "1-4_Day5_snap.oir").touch()
    (folder / "suite2p" / "plane0").mkdir(parents=True)
    (folder / "suite2p" / "plane0" / "registered.tif").touch()

    experiments = discover_experiments(tmp_path)

    assert len(experiments) == 1
    assert experiments[0].video.name == "1-4_Day5.tiff"
    assert experiments[0].still_tiff.name == "1-4_Day5_snap.tif"
    assert experiments[0].still_oir.name == "1-4_Day5_snap.oir"


def test_discover_experiments_records_oir_when_no_tiff_exists(tmp_path):
    folder = tmp_path / "1-2_Day3"
    folder.mkdir()
    (folder / "1-2_Day3.tiff").touch()
    (folder / "1-2_Day3_snap.oir").touch()

    experiment = discover_experiments(tmp_path)[0]

    assert experiment.still_tiff is None
    assert experiment.still_oir.name == "1-2_Day3_snap.oir"


def test_deep_merge_preserves_unmodified_nested_values():
    settings = {"run": {"do_registration": 1, "do_detection": True}}
    deep_merge(settings, {"run": {"do_registration": 2}})
    assert settings == {"run": {"do_registration": 2, "do_detection": True}}


def test_registration_configuration_uses_only_the_recording(tmp_path):
    folder = tmp_path / "recording"
    folder.mkdir()
    video = folder / "recording.tiff"
    video.touch()
    experiment = discover_experiments(tmp_path)[0]

    db, settings = BatchProcessor(suite2p_settings={"fs": 12.0})._build_registration_config(
        experiment
    )

    assert db["file_list"] == [video.name]
    assert db["save_folder"] == "suite2p_still"
    assert settings["fs"] == 12.0
    assert settings["run"]["do_detection"] is False
    assert settings["io"]["delete_bin"] is False
    assert settings["registration"]["reg_tif"] is False
    assert settings["registration"]["reg_tif_chan2"] is False


def test_batch_explicitly_skips_excluded_experiment(tmp_path):
    folder = tmp_path / "BP" / "1-4_Day2"
    folder.mkdir(parents=True)
    (folder / "1-4_Day2.tiff").touch()
    (folder / "1-4_Day2_snap.oir").touch()

    results = BatchProcessor(exclude_names={"1-4_Day2"}, dry_run=True).run(tmp_path)

    assert results[0]["status"] == "skipped_excluded"
