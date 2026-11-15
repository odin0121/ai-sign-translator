"""
Support Vector Machine (SVM) Classification Head
Implements multi-class RBF SVM (C=10.0, probability=True) over pooled 128-D latent vectors.
"""

import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)

# Default standard ASL Alphabet (26 letters) + 5 High-frequency WLASL phrases = 31 classes
DEFAULT_VOCABULARY = [
    # 26 Alphabet letters
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    # Dynamic words / phrases
    "HELLO", "THANK YOU", "YES", "NO", "PLEASE"
]


class SVMSignClassifier:
    """
    RBF Multi-Class SVM Classifier operating on 128-dimensional pooled latent embeddings.
    Provides deterministic maximum-margin classification and Platt scaled class probabilities.
    """

    def __init__(self, c_param=10.0, kernel='rbf', vocabulary=None, model_path=None,
                 gamma='scale', standardize=False):
        self.c_param = c_param
        self.kernel = kernel
        self.gamma = gamma
        self.standardize = standardize
        self.vocabulary = vocabulary or list(DEFAULT_VOCABULARY)
        self.model_path = model_path
        self.is_fitted = False
        self.training_source = None
        self.evaluation_metrics = None
        
        self.svm = SVC(
            kernel=self.kernel,
            C=self.c_param,
            probability=True,
            gamma=self.gamma,
            decision_function_shape='ovr',
            random_state=42
        )

        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def fit(self, X_latent, y_indices, training_source=None):
        """
        Fits multi-class SVM on pooled latent vectors.
        X_latent: np.ndarray shape (N, 128)
        y_indices: np.ndarray shape (N,) with integer class IDs
        """
        X_latent = np.asarray(X_latent, dtype=np.float32)
        y_indices = np.asarray(y_indices, dtype=int)

        logger.info(f"Training SVM on {X_latent.shape[0]} samples across {len(np.unique(y_indices))} classes...")
        if self.standardize:
            self.svm = make_pipeline(StandardScaler(), self.svm)
        self.svm.fit(X_latent, y_indices)
        self.is_fitted = True
        self.training_source = training_source
        logger.info("SVM training completed successfully.")
        return self

    def predict(self, latent_vector):
        """
        Inference on a single or batch of 128-D latent vectors.
        Returns:
            predicted_label (str), confidence (float), probabilities (dict or list)
        """
        latent = np.asarray(latent_vector, dtype=np.float32)
        if latent.ndim == 1:
            latent = np.expand_dims(latent, axis=0)

        if not self.is_fitted:
            raise RuntimeError("The SVM classifier has not been trained.")

        probs = self.svm.predict_proba(latent)[0]
        pred_idx = int(self.svm.predict(latent)[0])
        confidence = float(probs[np.flatnonzero(self.svm.classes_ == pred_idx)[0]])
        
        label = self.vocabulary[pred_idx] if pred_idx < len(self.vocabulary) else f"CLASS_{pred_idx}"
        
        # Top 3 candidates
        top3_indices = np.argsort(probs)[::-1][:3]
        top3_candidates = [
            (self.vocabulary[idx] if idx < len(self.vocabulary) else f"CLASS_{idx}", float(probs[idx]))
            for idx in top3_indices
        ]

        return label, confidence, top3_candidates

    def evaluate(self, X_test, y_test):
        """Evaluates SVM on holdout dataset."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        probabilities = self.svm.predict_proba(X_test)
        # SVC's calibrated probabilities can reorder the class with the largest
        # margin, especially for many classes. Use the SVM decision rule for
        # top-1 and retain calibrated probabilities for confidence/ranking.
        preds = self.svm.predict(X_test)
        acc = accuracy_score(y_test, preds)
        matrix = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        
        return {
            "accuracy": float(acc),
            "confusion_matrix": matrix.tolist(),
            "report": report
        }

    def save(self, filepath):
        """Persists trained SVM and class vocabulary using joblib."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "svm": self.svm,
            "vocabulary": self.vocabulary,
            "is_fitted": self.is_fitted,
            "training_source": self.training_source,
            "evaluation_metrics": self.evaluation_metrics,
            "c_param": self.c_param,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "standardize": self.standardize
        }
        joblib.dump(payload, filepath)
        logger.info(f"SVM model saved to {filepath}")

    def load(self, filepath):
        """Loads persisted SVM and vocabulary."""
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return False

        payload = joblib.load(filepath)
        if isinstance(payload, dict):
            self.svm = payload.get("svm", self.svm)
            self.vocabulary = payload.get("vocabulary", self.vocabulary)
            self.is_fitted = payload.get("is_fitted", True)
            self.training_source = payload.get("training_source")
            self.evaluation_metrics = payload.get("evaluation_metrics")
            self.c_param = payload.get("c_param", 10.0)
            self.kernel = payload.get("kernel", "rbf")
            self.gamma = payload.get("gamma", "scale")
            self.standardize = payload.get("standardize", False)
        else:
            self.svm = payload
            self.is_fitted = True
            self.training_source = None
            self.evaluation_metrics = None
            
        logger.info(f"SVM loaded successfully from {filepath}")
        return True
