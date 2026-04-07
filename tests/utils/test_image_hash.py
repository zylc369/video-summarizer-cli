"""Tests for image_hash utility module."""

import numpy as np
from imagehash import ImageHash
from PIL import Image

from src.utils.image_hash import are_similar, compute_phash, deduplicate_frames


def test_compute_phash_returns_hash(tmp_path, mocker):
    """Verify compute_phash returns an ImageHash object."""
    mock_image = mocker.MagicMock(spec=Image.Image)
    mocker.patch("PIL.Image.open", return_value=mock_image)

    expected_hash = ImageHash(np.zeros((8, 8), dtype=bool))
    mocker.patch("imagehash.phash", return_value=expected_hash)

    image_file = tmp_path / "test.jpg"
    image_file.write_text("dummy")

    result = compute_phash(image_file)

    assert isinstance(result, ImageHash)
    assert np.array_equal(result.hash, expected_hash.hash)


def test_are_similar_identical_images():
    """Test that identical hashes return True."""
    hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
    hash2 = ImageHash(np.zeros((8, 8), dtype=bool))

    result = are_similar(hash1, hash2, threshold=0.95)

    assert result is True


def test_are_similar_different_images():
    """Test that completely different hashes return False."""
    hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
    hash2 = ImageHash(np.ones((8, 8), dtype=bool))

    result = are_similar(hash1, hash2, threshold=0.95)

    assert result is False


def test_are_similar_threshold_boundary():
    """Test behavior exactly at the threshold boundary."""
    # For 8x8 hash, max distance is 64. To get exactly 0.95 similarity,
    # we need distance = 64 * (1 - 0.95) = 3.2. With integer distances,
    # distance=3 gives similarity=1-(3/64)=0.953125 >= 0.95 (True),
    # while threshold=0.954 gives similarity < threshold (False).

    hash1 = ImageHash(np.zeros((8, 8), dtype=bool))
    hash2_array = np.zeros((8, 8), dtype=bool)
    hash2_array[0, 0] = True
    hash2_array[0, 1] = True
    hash2_array[0, 2] = True
    hash2 = ImageHash(hash2_array)

    result = are_similar(hash1, hash2, threshold=0.95)
    assert result is True

    result = are_similar(hash1, hash2, threshold=0.954)
    assert result is False


def test_deduplicate_frames_removes_duplicates(tmp_path, mocker):
    """Test that duplicate frames are removed."""
    # Hash sequence: A, A (dup), B, B (dup), C (distinct)
    hash_a = ImageHash(np.zeros((8, 8), dtype=bool))
    hash_b = ImageHash(np.ones((8, 8), dtype=bool))
    hash_c = ImageHash(np.zeros((8, 8), dtype=bool))
    hash_c.hash[0:4, 0:4] = True  # Make hash_c distinct from A and B

    hash_sequence = [hash_a, hash_a, hash_b, hash_b, hash_c]
    mock_compute_phash = mocker.patch(
        "src.utils.image_hash.compute_phash", side_effect=hash_sequence
    )

    frame_paths = [tmp_path / f"frame_{i}.jpg" for i in range(5)]
    for path in frame_paths:
        path.write_text("dummy")

    result = deduplicate_frames(frame_paths, threshold=0.95)

    # Verify frames 0, 2, 4 kept (3 frames total, duplicates removed)
    assert len(result) == 3
    assert result == [frame_paths[0], frame_paths[2], frame_paths[4]]
    assert mock_compute_phash.call_count == 5


def test_deduplicate_frames_keeps_first(tmp_path, mocker):
    """Test that the first frame is always kept."""
    hash_a = ImageHash(np.zeros((8, 8), dtype=bool))
    hash_b = ImageHash(np.ones((8, 8), dtype=bool))
    hash_c = ImageHash(np.zeros((8, 8), dtype=bool))
    hash_c.hash[0:4, 0:4] = True

    hash_sequence = [hash_a, hash_a, hash_a, hash_b, hash_c]
    mocker.patch("src.utils.image_hash.compute_phash", side_effect=hash_sequence)

    frame_paths = [tmp_path / f"frame_{i}.jpg" for i in range(5)]
    for path in frame_paths:
        path.write_text("dummy")

    result = deduplicate_frames(frame_paths, threshold=0.95)

    assert len(result) >= 1
    assert result[0] == frame_paths[0]


def test_deduplicate_frames_preserves_order(tmp_path, mocker):
    """Test that output order matches input order."""
    hashes = []
    for i in range(5):
        h = ImageHash(np.zeros((8, 8), dtype=bool))
        h.hash[i, :] = True
        hashes.append(h)

    mocker.patch("src.utils.image_hash.compute_phash", side_effect=hashes)

    frame_paths = [tmp_path / f"frame_{i}.jpg" for i in range(5)]
    for path in frame_paths:
        path.write_text("dummy")

    result = deduplicate_frames(frame_paths, threshold=0.95)

    assert result == frame_paths


def test_deduplicate_frames_empty_list():
    """Test that empty input returns empty output."""
    result = deduplicate_frames([], threshold=0.95)
    assert result == []


def test_deduplicate_frames_single_frame(tmp_path, mocker):
    """Test that single frame returns single frame."""
    hash_a = ImageHash(np.zeros((8, 8), dtype=bool))
    mocker.patch("src.utils.image_hash.compute_phash", return_value=hash_a)

    frame_path = tmp_path / "frame_0.jpg"
    frame_path.write_text("dummy")

    result = deduplicate_frames([frame_path], threshold=0.95)

    assert len(result) == 1
    assert result[0] == frame_path
