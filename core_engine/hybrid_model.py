"""
Hybrid TCN + Multi-Head Attention Transformer Backbone
Implements dilated causal 1D convolutions + Multi-Head Self-Attention
for kinematic sign sequence encoding.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_backbone(T=60, D=63, latent_dim=128):
    """
    Constructs the exact TCN + Transformer backbone specified in the implementation plan.
    
    Architecture:
      Input: (Batch, T=60, D=63)
      Conv1D(64, k=3, dilation=1, causal) -> BatchNorm
      Conv1D(128, k=3, dilation=2, causal) -> BatchNorm
      Conv1D(latent_dim=128, k=3, dilation=4, causal)
      MultiHeadAttention(heads=4, key_dim=32) + Residual + LayerNorm
      FFN(Dense(256) -> Dense(128)) + Residual + LayerNorm
      GlobalAveragePooling1D -> (Batch, latent_dim=128)
    """
    inp = layers.Input(shape=(T, D), name="kinematic_sequence_input")
    
    # Dilated Causal Convolutional Hierarchy (TCN)
    x = layers.Conv1D(64, 3, dilation_rate=1, padding='causal', activation='relu', name="tcn_conv1")(inp)
    x = layers.BatchNormalization(name="tcn_bn1")(x)
    
    x = layers.Conv1D(128, 3, dilation_rate=2, padding='causal', activation='relu', name="tcn_conv2")(x)
    x = layers.BatchNormalization(name="tcn_bn2")(x)
    
    x = layers.Conv1D(latent_dim, 3, dilation_rate=4, padding='causal', activation='relu', name="tcn_conv3")(x)
    
    # Multi-Head Self-Attention Block
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="transformer_mha")(x, x)
    x = layers.LayerNormalization(name="attn_layernorm")(layers.Add(name="attn_residual")([x, attn]))
    
    # Position-Wise Feed Forward Network (FFN)
    ffn = layers.Dense(256, activation='relu', name="ffn_dense1")(x)
    ffn = layers.Dense(latent_dim, activation='relu', name="ffn_dense2")(ffn)
    x = layers.LayerNormalization(name="ffn_layernorm")(layers.Add(name="ffn_residual")([x, ffn]))
    
    # Temporal Latent Pooling
    out = layers.GlobalAveragePooling1D(name="latent_pooling")(x)
    
    backbone = models.Model(inp, out, name="TCN_Transformer")
    return backbone


class HybridTCNTransformer:
    """
    Wrapper for model lifecycle: building, training, latent feature extraction,
    and weights persistence.
    """
    def __init__(self, T=60, D=63, latent_dim=128, num_classes=30, weights_path=None):
        self.T = T
        self.D = D
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.weights_path = weights_path
        
        self.backbone = build_backbone(T=self.T, D=self.D, latent_dim=self.latent_dim)
        self.classifier_model = self._build_classifier()
        
        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def _build_classifier(self):
        """Builds an end-to-end model for pretraining the backbone features."""
        inp = self.backbone.input
        latent = self.backbone.output
        drop = layers.Dropout(0.3)(latent)
        logits = layers.Dense(self.num_classes, activation='softmax', name="softmax_head")(drop)
        model = models.Model(inp, logits, name="EndToEnd_SignClassifier")
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def extract_latent(self, seq_window):
        """
        Extract pooled 128-D latent vector from a sequence window.
        Input shape: (T, D) or (1, T, D) or (N, T, D)
        Returns: np.ndarray of shape (N, 128)
        """
        seq = np.asarray(seq_window, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)
        
        tensor_seq = tf.convert_to_tensor(seq, dtype=tf.float32)
        latent = self.backbone(tensor_seq, training=False)
        return latent.numpy()

    def fit_backbone(self, X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, validation_data=None):
        """Trains the backbone end-to-end on sequence tensors."""
        kwargs = {"epochs": epochs, "batch_size": batch_size, "verbose": 1}
        if validation_data is not None:
            kwargs["validation_data"] = validation_data
        else:
            kwargs["validation_split"] = validation_split

        history = self.classifier_model.fit(X_train, y_train, **kwargs)
        return history

    def save_weights(self, filepath):
        """Saves backbone weights to an HDF5 or Keras file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Save backbone weights
        self.backbone.save_weights(filepath)

    def load_weights(self, filepath):
        """Loads backbone weights with dual format tolerance."""
        candidates = [filepath]
        if filepath.endswith('.h5') and not filepath.endswith('.weights.h5'):
            candidates.insert(0, filepath.replace('.h5', '.weights.h5'))
        elif filepath.endswith('.weights.h5'):
            candidates.append(filepath.replace('.weights.h5', '.h5'))

        for p in candidates:
            if os.path.exists(p):
                try:
                    self.backbone.load_weights(p)
                    return True
                except Exception:
                    continue
        return False


class HybridTranslator:
    """
    Blueprint Implementation from Agent Implementation Plan Sheet (Section 4, #3):
    Coordinates backbone latent extraction, TensorFlow gradient-tape XAI saliency,
    and multi-class SVM inference in a single call.
    """
    def __init__(self, backbone, svm_model=None):
        from sklearn.svm import SVC
        self.backbone = backbone
        self.svm = svm_model or SVC(kernel='rbf', C=10.0, probability=True)

    def predict_with_xai(self, seq_window):
        t_seq = tf.convert_to_tensor(seq_window, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(t_seq)
            latent = self.backbone(t_seq, training=False)
            target = tf.reduce_max(latent)
        grads = tape.gradient(target, t_seq)
        if grads is not None:
            joint_count = int(t_seq.shape[-1]) // 3
            saliency = (
                tf.reduce_mean(tf.abs(grads), axis=[0, 1])
                .numpy()
                .reshape(joint_count, 3)
                .sum(axis=-1)
            )
            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        else:
            joint_count = int(t_seq.shape[-1]) // 3
            saliency = np.ones(joint_count, dtype=np.float32) / joint_count
        pred = self.svm.predict(latent.numpy())[0]
        prob = np.max(self.svm.predict_proba(latent.numpy())[0])
        return pred, prob, saliency
