"""Perceptual hash utilities for keyframe deduplication."""

from pathlib import Path
from typing import TYPE_CHECKING

import imagehash
from PIL import Image

if TYPE_CHECKING:
    from imagehash import ImageHash


def compute_phash(image_path: Path, hash_size: int = 8) -> "ImageHash":
    """Compute perceptual hash for an image.

    Args:
        image_path: Path to the image file
        hash_size: Size of the hash (default 8, resulting in 64-bit hash)

    Returns:
        ImageHash object representing the perceptual hash

    Raises:
        FileNotFoundError: If image file doesn't exist
        PIL.UnidentifiedImageError: If file is not a valid image
    """
    image = Image.open(image_path)
    return imagehash.phash(image, hash_size)


def are_similar(
    hash1: "ImageHash",
    hash2: "ImageHash",
    threshold: float = 0.95,
) -> bool:
    """Check if two perceptual hashes are similar.

    Args:
        hash1: First image hash
        hash2: Second image hash
        threshold: Similarity threshold (0.0 to 1.0, default 0.95)

    Returns:
        True if hashes are similar (similarity >= threshold), False otherwise
    """
    hamming_distance = hash1 - hash2
    # For hash_size=8, max distance is 64 (8x8 bits)
    hash_size = hash1.hash.shape[0] if hasattr(hash1, "hash") else 8
    max_distance = hash_size * hash_size

    similarity = 1.0 - (hamming_distance / max_distance)
    return bool(similarity >= threshold)


def deduplicate_frames(
    frame_paths: list[Path],
    threshold: float = 0.95,
) -> list[Path]:
    """Remove duplicate keyframes using perceptual hashing.

    Keeps only frames that are visually distinct from the previously kept frame.
    The first frame is always kept. Frames are compared in order.

    Args:
        frame_paths: Sorted list of keyframe image paths
        threshold: Similarity threshold (0.0 to 1.0, default 0.95)

    Returns:
        Deduplicated list of frame paths in original order
    """
    if not frame_paths:
        return []

    deduplicated: list[Path] = []
    last_hash: "ImageHash | None" = None

    for frame_path in frame_paths:
        current_hash = compute_phash(frame_path)

        if last_hash is None:
            deduplicated.append(frame_path)
            last_hash = current_hash
        elif not are_similar(last_hash, current_hash, threshold):
            deduplicated.append(frame_path)
            last_hash = current_hash

    return deduplicated
