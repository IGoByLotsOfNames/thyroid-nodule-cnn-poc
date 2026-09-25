from __future__ import annotations

import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def stratified_directory_split(
    source: Path,
    destination: Path,
    *,
    train: float = 0.70,
    validation: float = 0.15,
    seed: int = 42,
) -> None:
    """Copy class directories into train/validation/test splits.

    The source must contain one child directory per class. Files are copied,
    so the source dataset remains unchanged.
    """
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError("Split ratios must leave a non-empty test proportion")
    rng = random.Random(seed)
    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        images = [path for path in class_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
        rng.shuffle(images)
        train_end = int(len(images) * train)
        validation_end = train_end + int(len(images) * validation)
        splits = {
            "train": images[:train_end],
            "validation": images[train_end:validation_end],
            "test": images[validation_end:],
        }
        for split, paths in splits.items():
            output = destination / split / class_dir.name
            output.mkdir(parents=True, exist_ok=True)
            for path in paths:
                shutil.copy2(path, output / path.name)


def image_datasets(root: Path, image_size: tuple[int, int], batch_size: int, seed: int):
    import tensorflow as tf

    common = dict(image_size=image_size, batch_size=batch_size, label_mode="binary")
    train = tf.keras.utils.image_dataset_from_directory(
        root / "train", shuffle=True, seed=seed, **common
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        root / "validation", shuffle=False, **common
    )
    test = tf.keras.utils.image_dataset_from_directory(root / "test", shuffle=False, **common)
    return train, validation, test

