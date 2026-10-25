"""Behavior tests for carclass.dataset (webp glob + PKSampler)."""

from pathlib import Path

from carclass.dataset import PKSampler, VehicleDataset


def _make_img(path: Path, fmt: str = "JPEG", color=(10, 20, 30)):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, fmt)


def test_imports_under_torch():
    import torch  # noqa: F401 -- proves torch present for this module's stack

    assert VehicleDataset is not None
    assert PKSampler is not None


def test_webp_only_class_lists_samples(tmp_path):
    _make_img(tmp_path / "sedan" / "a.jpg")
    _make_img(tmp_path / "webp_only" / "b.webp", fmt="WEBP")
    ds = VehicleDataset(tmp_path)
    assert "webp_only" in ds.classes
    by_class = {}
    for _, label in ds.samples:
        by_class[ds.classes[label]] = by_class.get(ds.classes[label], 0) + 1
    assert by_class.get("webp_only", 0) == 1


def test_pksampler_drops_small_groups_with_log(capsys):
    groups = {0: [0, 1, 2, 3], 1: [4]}  # group 1 has < K=2 shots
    sampler = PKSampler(groups, P=2, K=2, iterations=2)
    out = capsys.readouterr().out
    assert "dropping 1 groups" in out
    assert 1 not in sampler.group_to_indices
    assert set(sampler.group_to_indices) == {0}
    batches = list(iter(sampler))
    assert len(batches) == 2
    assert all(len(b) == 2 for b in batches)
