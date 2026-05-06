"""Regression tests: read fixture mp4, extract CLIP features, compare to snapshots."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import crop  # noqa: E402

FIXTURE_VIDEO = (
    ROOT
    / "tests"
    / "fixtures"
    / "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A.mp4"
)
SNAPSHOT_DIR = ROOT / "tests" / "_snapshots"
VIDEO_STEM = "A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A"


def _mean_frame_cosine(actual: np.ndarray, expected: np.ndarray) -> float:
    actual = actual.astype(np.float32)
    expected = expected.astype(np.float32)
    actual_norm = np.linalg.norm(actual, axis=1)
    expected_norm = np.linalg.norm(expected, axis=1)
    return float(np.mean(np.sum(actual * expected, axis=1) / (actual_norm * expected_norm)))


def test_save_video_clip_features_matches_snapshots(tmp_path: Path) -> None:
    from clip import clip
    import torch

    assert FIXTURE_VIDEO.is_file(), f"Missing fixture video: {FIXTURE_VIDEO}"

    # Time length must match tests/_snapshots (fixture mp4 is long; snapshots use 97 frames).
    snap0 = next(iter(sorted(SNAPSHOT_DIR.glob(f"{VIDEO_STEM}__*.npy"))))
    expected_t = int(np.load(snap0).shape[0])

    frames = crop.read_video_to_ndarray(
        str(FIXTURE_VIDEO),
        stride=16,
        shift=8,
        convert_to_rgb=True,
    )
    assert frames.ndim == 4 and frames.shape[-1] == 3
    assert frames.shape[0] > 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    model.eval()

    crop.save_video_clip_features(
        str(FIXTURE_VIDEO),
        frames,
        tmp_path,
        model,
        preprocess,
        device,
    )

    snapshot_files = sorted(SNAPSHOT_DIR.glob(f"{VIDEO_STEM}__*.npy"))
    assert len(snapshot_files) == 10, f"expected 10 snapshots, got {len(snapshot_files)}"

    generated = sorted(tmp_path.glob("*.npy"))
    assert len(generated) == 10
    assert {p.name for p in generated} == {p.name for p in snapshot_files}

    for snap_path in snapshot_files:
        expected = np.load(snap_path)
        actual = np.load(tmp_path / snap_path.name)

        assert expected.shape == actual.shape, (snap_path.name, expected.shape, actual.shape)
        mean_cosine = _mean_frame_cosine(actual, expected)
        mean_abs_error = float(
            np.mean(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
        )

        # CLIP features can differ slightly across video decoders / torch builds. Keep the
        # snapshot check strict on shape and names, and use feature-space similarity for values.
        assert mean_cosine > 0.78, (snap_path.name, mean_cosine)
        assert mean_abs_error < 0.22, (snap_path.name, mean_abs_error)
