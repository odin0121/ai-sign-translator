"""Train the separate Indian Sign Language emergency-sign model."""

import argparse
import logging
import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("KERAS_HOME", "/tmp/keras")

import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping

from core_engine.data_loader import SignDataLoader
from core_engine.hybrid_model import HybridTCNTransformer
from core_engine.landmark_extractor import LANDMARK_FEATURE_VERSION
from core_engine.svm_classifier import SVMSignClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ISLTrainer")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ISL emergency-sign model using signer-independent splits."
    )
    parser.add_argument(
        "--archive",
        default=os.path.join(
            BASE_DIR,
            "..",
            "A Video Dataset of the Hand Gestures of Indian Sign Language Words used in Emergency Situations.zip",
        ),
        help="Path to the uploaded Indian Sign Language emergency video archive.",
    )
    parser.add_argument(
        "--cache",
        default=os.path.join(BASE_DIR, "weights", "isl_emergency_landmarks.npz"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--augmentations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.augmentations < 0:
        raise ValueError("epochs/batch-size must be positive and augmentations cannot be negative.")
    if not os.path.isfile(args.archive):
        raise FileNotFoundError(f"ISL emergency archive not found: {args.archive}")

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    weights_dir = os.path.join(BASE_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    backbone_path = os.path.join(weights_dir, "isl_tcn_transformer.weights.h5")
    svm_path = os.path.join(weights_dir, "isl_svm_model.joblib")

    loader = SignDataLoader(sequence_length=60, feature_dim=63)
    X_train, y_train, X_val, y_val, X_test, y_test, vocabulary = (
        loader.load_isl_emergency_archive(
            args.archive,
            cache_path=args.cache,
            seed=args.seed,
        )
    )
    if len(np.unique(y_train)) != len(vocabulary):
        raise ValueError("Every ISL label must have usable training clips.")
    if len(np.unique(y_val)) != len(vocabulary) or len(np.unique(y_test)) != len(vocabulary):
        raise ValueError("Every ISL label must be represented in validation and test splits.")
    logger.info(
        "ISL data ready: %d train, %d validation, %d test clips across %d labels.",
        len(y_train),
        len(y_val),
        len(y_test),
        len(vocabulary),
    )

    X_fit_parts = [X_train]
    y_fit_parts = [y_train]
    for augmentation_index in range(args.augmentations):
        logger.info(
            "Creating augmented training sequences (%d/%d).",
            augmentation_index + 1,
            args.augmentations,
        )
        X_fit_parts.append(
            np.asarray(
                [loader.augment_sequence(sequence) for sequence in X_train],
                dtype=np.float32,
            )
        )
        y_fit_parts.append(y_train.copy())
    X_fit = np.concatenate(X_fit_parts, axis=0)
    y_fit = np.concatenate(y_fit_parts, axis=0)
    class_ids = np.unique(y_fit)
    class_weights = dict(
        zip(
            class_ids.tolist(),
            compute_class_weight(class_weight="balanced", classes=class_ids, y=y_fit).tolist(),
        )
    )

    model = HybridTCNTransformer(
        T=60,
        D=63,
        latent_dim=128,
        num_classes=len(vocabulary),
    )
    model.classifier_model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=8,
                restore_best_weights=True,
            )
        ],
        class_weight=class_weights,
        verbose=1,
    )
    model.save_weights(backbone_path)

    train_latent = model.extract_latent(X_train)
    val_latent = model.extract_latent(X_val)
    test_latent = model.extract_latent(X_test)
    classifier = SVMSignClassifier(c_param=10.0, kernel="rbf", vocabulary=vocabulary)
    classifier.fit(
        train_latent,
        y_train,
        training_source="isl_emergency_v1",
    )

    val_probs = classifier.svm.predict_proba(val_latent)
    val_predictions = classifier.svm.predict(val_latent)
    val_confidence = np.max(val_probs, axis=1)
    target_coverage = 0.70
    candidates = []
    for threshold in np.arange(0.01, 0.951, 0.01):
        accepted = val_confidence >= threshold
        coverage = float(np.mean(accepted))
        if coverage >= target_coverage:
            precision = float(np.mean(val_predictions[accepted] == y_val[accepted]))
            candidates.append((precision, coverage, float(threshold)))
    if not candidates:
        raise ValueError("Validation split cannot support confidence-threshold calibration.")
    validation_precision, validation_coverage, min_confidence = max(
        candidates,
        key=lambda item: (item[0], item[1], -item[2]),
    )

    results = classifier.evaluate(test_latent, y_test)
    test_probs = classifier.svm.predict_proba(test_latent)
    test_predictions = classifier.svm.predict(test_latent)
    accepted_test = np.max(test_probs, axis=1) >= min_confidence
    test_coverage = float(np.mean(accepted_test))
    test_selective_accuracy = (
        float(np.mean(test_predictions[accepted_test] == y_test[accepted_test]))
        if np.any(accepted_test)
        else None
    )
    logger.info("Held-out signer-independent ISL top-1 accuracy: %.2f%%", results["accuracy"] * 100)
    logger.info(
        "Accepted test clips: %d/%d (%.1f%% coverage), accuracy %s; cutoff %.2f.",
        int(np.sum(accepted_test)),
        len(y_test),
        test_coverage * 100,
        f"{test_selective_accuracy * 100:.2f}%" if test_selective_accuracy is not None else "unavailable",
        min_confidence,
    )
    logger.info("Held-out classification report:\n%s", results["report"])
    classifier.evaluation_metrics = {
        "landmark_feature_version": LANDMARK_FEATURE_VERSION,
        "test_accuracy": results["accuracy"],
        "min_confidence": min_confidence,
        "validation_selective_accuracy": validation_precision,
        "validation_coverage": validation_coverage,
        "test_selective_accuracy": test_selective_accuracy,
        "test_coverage": test_coverage,
        "test_samples": int(len(y_test)),
        "validation_samples": int(len(y_val)),
        "classes": int(len(vocabulary)),
        "split_strategy": "signer_independent",
        "split_seed": args.seed,
    }
    classifier.save(svm_path)
    logger.info("Saved ISL backbone: %s", backbone_path)
    logger.info("Saved ISL classifier: %s", svm_path)


if __name__ == "__main__":
    main()
