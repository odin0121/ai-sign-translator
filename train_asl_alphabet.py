"""Train a static ASL alphabet classifier from the uploaded alphabet image archive."""

import argparse
import logging
import os
import random
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core_engine.landmark_extractor import LANDMARK_FEATURE_VERSION, LandmarkExtractor
from core_engine.svm_classifier import SVMSignClassifier

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CACHE_VERSION = 3
logger = logging.getLogger("ASLAlphabetTrainer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an A-Z hand-landmark classifier from the ASL alphabet image archive."
    )
    parser.add_argument(
        "--archive",
        default=str(BASE_DIR.parent / "archive.zip"),
        help="ZIP archive with Gesture Image Data/<letter>/*.jpg folders.",
    )
    parser.add_argument(
        "--cache",
        default=str(BASE_DIR / "weights" / "asl_alphabet_landmarks.npz"),
        help="Path for extracted train/validation landmark features.",
    )
    parser.add_argument(
        "--model",
        default=str(BASE_DIR / "weights" / "asl_alphabet_svm.joblib"),
        help="Path for the trained alphabet classifier.",
    )
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _archive_members(archive):
    members = {letter: [] for letter in LETTERS}
    for member in archive.namelist():
        parts = member.split("/")
        if (
            len(parts) == 3
            and parts[0] == "Gesture Image Data"
            and parts[1] in members
            and parts[2].lower().endswith((".jpg", ".jpeg", ".png"))
        ):
            members[parts[1]].append(member)
    for class_members in members.values():
        class_members.sort()
    return members


def _cache_matches(cache_path, archive_path, samples_per_class, seed):
    if not os.path.isfile(cache_path):
        return False
    stat = os.stat(archive_path)
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            return (
                str(Path(archive_path).resolve()) == str(cache["source_path"].item())
                and stat.st_size == int(cache["source_size"])
                and stat.st_mtime_ns == int(cache["source_mtime_ns"])
                and int(cache["landmark_feature_version"]) == LANDMARK_FEATURE_VERSION
                and int(cache["cache_version"]) == CACHE_VERSION
                and cache["vocabulary"].astype(str).tolist() == LETTERS
                and int(cache["samples_per_class"]) == samples_per_class
                and int(cache["seed"]) == seed
            )
    except (OSError, KeyError, ValueError):
        return False


def _extract_features(archive_path, cache_path, samples_per_class, seed):
    archive_path = str(Path(archive_path).resolve())
    if _cache_matches(cache_path, archive_path, samples_per_class, seed):
        logger.info("Using cached alphabet landmark features: %s", cache_path)
        with np.load(cache_path, allow_pickle=False) as cache:
            return (
                cache["X_train"],
                cache["y_train"],
                cache["X_val"],
                cache["y_val"],
                cache["demo_members"].astype(str).tolist(),
            )

    if samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2.")
    if not os.path.isfile(archive_path):
        raise FileNotFoundError(f"ASL alphabet archive not found: {archive_path}")

    rng = random.Random(seed)
    class_samples = {}
    demo_members = []
    extractor = LandmarkExtractor(static_image_mode=True, min_detection_confidence=0.2)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            classes = _archive_members(archive)
            missing = [letter for letter, members in classes.items() if not members]
            if missing:
                raise ValueError(
                    "ASL alphabet archive is missing image folders for: "
                    + ", ".join(missing)
                )

            for class_index, letter in enumerate(LETTERS):
                members = classes[letter]
                rng.shuffle(members)
                usable_samples = []
                for member in members:
                    encoded = np.frombuffer(archive.read(member), dtype=np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    points = extractor.extract(image)
                    if not extractor.has_detection or not np.isfinite(points).all() or not np.any(points):
                        continue
                    usable_samples.append((points.astype(np.float32, copy=False), member))
                    if len(usable_samples) == samples_per_class:
                        break
                class_samples[letter] = usable_samples
                logger.info(
                    "Found %d usable examples for %s (requested up to %d).",
                    len(usable_samples),
                    letter,
                    samples_per_class,
                )
    finally:
        if extractor.neural_detector is not None:
            extractor.neural_detector.close()
        if extractor.hands is not None:
            extractor.hands.close()

    balanced_count = min(len(class_samples[letter]) for letter in LETTERS)
    if balanced_count < 3:
        counts = ", ".join(
            f"{letter}: {len(class_samples[letter])}"
            for letter in LETTERS
            if len(class_samples[letter]) == balanced_count
        )
        raise ValueError(
            f"Only {balanced_count} usable examples are available for the least-detected "
            f"letter(s) ({counts}); at least 3 are needed to create train and validation splits."
        )

    train_features, train_labels = [], []
    val_features, val_labels = [], []
    val_count = max(1, round(balanced_count * 0.2))
    for class_index, letter in enumerate(LETTERS):
        samples = class_samples[letter][:balanced_count]
        validation_samples = samples[:val_count]
        training_samples = samples[val_count:]
        val_features.extend(points for points, _member in validation_samples)
        val_labels.extend([class_index] * len(validation_samples))
        train_features.extend(points for points, _member in training_samples)
        train_labels.extend([class_index] * len(training_samples))
        demo_members.append(training_samples[0][1])
        logger.info(
            "Balanced %s to %d usable examples (%d train, %d validation).",
            letter,
            balanced_count,
            len(training_samples),
            len(validation_samples),
        )

    archive_stat = os.stat(archive_path)
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_train=np.asarray(train_features, dtype=np.float32),
        y_train=np.asarray(train_labels, dtype=np.int32),
        X_val=np.asarray(val_features, dtype=np.float32),
        y_val=np.asarray(val_labels, dtype=np.int32),
        vocabulary=np.asarray(LETTERS, dtype=np.str_),
        demo_members=np.asarray(demo_members, dtype=np.str_),
        source_path=np.asarray(archive_path, dtype=np.str_),
        source_size=np.asarray(archive_stat.st_size, dtype=np.int64),
        source_mtime_ns=np.asarray(archive_stat.st_mtime_ns, dtype=np.int64),
        landmark_feature_version=np.asarray(LANDMARK_FEATURE_VERSION, dtype=np.int32),
        cache_version=np.asarray(CACHE_VERSION, dtype=np.int32),
        samples_per_class=np.asarray(samples_per_class, dtype=np.int32),
        seed=np.asarray(seed, dtype=np.int32),
    )
    return (
        np.asarray(train_features, dtype=np.float32),
        np.asarray(train_labels, dtype=np.int32),
        np.asarray(val_features, dtype=np.float32),
        np.asarray(val_labels, dtype=np.int32),
        demo_members,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    if args.samples_per_class < 2:
        raise ValueError("--samples-per-class must be at least 2.")

    X_train, y_train, X_val, y_val, _demo_members = _extract_features(
        args.archive,
        args.cache,
        args.samples_per_class,
        args.seed,
    )
    classifier = SVMSignClassifier(
        c_param=10.0,
        kernel="rbf",
        vocabulary=LETTERS,
    )
    logger.info(
        "Training A-Z classifier with %d train and %d validation hand samples.",
        len(y_train),
        len(y_val),
    )
    classifier.fit(X_train, y_train, training_source="asl_alphabet_v1")

    probabilities = classifier.svm.predict_proba(X_val)
    predictions = classifier.svm.classes_[np.argmax(probabilities, axis=1)]
    confidence = np.max(probabilities, axis=1)
    accuracy = float(accuracy_score(y_val, predictions))
    threshold_candidates = []
    for threshold in np.arange(0.25, 0.96, 0.05):
        accepted = confidence >= threshold
        if np.count_nonzero(accepted) >= max(26, round(len(y_val) * 0.1)):
            threshold_candidates.append(
                (float(np.mean(predictions[accepted] == y_val[accepted])),
                 float(np.mean(accepted)),
                 float(threshold))
            )
    # Keep at least 95% of validation predictions available for live practice.
    responsive_candidates = [
        candidate for candidate in threshold_candidates if candidate[1] >= 0.95
    ]
    min_confidence = max(
        responsive_candidates or threshold_candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )[2] if threshold_candidates else 0.25
    accepted = confidence >= min_confidence
    classifier.evaluation_metrics = {
        "landmark_feature_version": LANDMARK_FEATURE_VERSION,
        "test_accuracy": accuracy,
        "validation_accuracy": accuracy,
        "validation_samples": int(len(y_val)),
        "validation_coverage": float(np.mean(accepted)),
        "validation_selective_accuracy": (
            float(np.mean(predictions[accepted] == y_val[accepted]))
            if np.any(accepted) else None
        ),
        "min_confidence": min_confidence,
        "classes": len(LETTERS),
        "training_samples": int(len(y_train)),
    }
    classifier.save(args.model)
    logger.info("Validation accuracy: %.2f%%", accuracy * 100)
    logger.info(
        "Validation report:\n%s",
        classification_report(y_val, predictions, labels=range(len(LETTERS)), target_names=LETTERS, zero_division=0),
    )
    logger.info("Saved alphabet classifier: %s", args.model)
    logger.info("Saved extracted features: %s", args.cache)


if __name__ == "__main__":
    main()