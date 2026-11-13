"""
Post-Hoc Explainable AI (XAI) Saliency Explainer
Computes gradient-weighted joint attributions:
  A_k = sum_{c in {x,y,z}} | d(max(z)) / d(p_{k,c}) |
and dynamically renders sleek cyberpunk/medical thermal skeleton overlays,
dynamic HUD telemetry, and targeting brackets onto camera frames.
"""

import cv2
import numpy as np
import tensorflow as tf
from .landmark_extractor import JOINT_NAMES, HAND_CONNECTIONS


class XAISaliencyExplainer:
    """
    Computes joint-level saliency attribution via TensorFlow gradient backpropagation
    with temporal EMA smoothing and renders eye-pleasing real-time attribution overlays.
    """

    def __init__(self, backbone, svm_classifier, ema_alpha=0.35):
        self.backbone = backbone
        self.svm = svm_classifier
        self.ema_alpha = ema_alpha
        self.smoothed_saliency = None
        self.pulse_phase = 0.0

    def compute_saliency(self, seq_window):
        """
        Calculates gradient-based saliency for the 21 hand joints across a sequence window.
        Applies Exponential Moving Average (EMA) to eliminate frame-to-frame thermal flicker.
        """
        seq = np.asarray(seq_window, dtype=np.float32)
        if seq.ndim == 2:
            seq = np.expand_dims(seq, axis=0)

        t_seq = tf.convert_to_tensor(seq, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(t_seq)
            latent = self.backbone(t_seq, training=False)
            target = tf.reduce_max(latent)

        grads = tape.gradient(target, t_seq)

        if grads is None:
            joint_count = seq.shape[-1] // 3
            raw_saliency = np.ones(joint_count, dtype=np.float32) / joint_count
        else:
            abs_grads = tf.reduce_mean(tf.abs(grads), axis=[0, 1]).numpy()
            raw_saliency = abs_grads.reshape(-1, len(JOINT_NAMES), 3).sum(axis=-1).flatten()
            s_min = float(np.min(raw_saliency))
            s_max = float(np.max(raw_saliency))
            if s_max - s_min > 1e-6:
                raw_saliency = (raw_saliency - s_min) / (s_max - s_min)
            else:
                raw_saliency = np.ones(21, dtype=np.float32) * 0.5

        # Temporal EMA smoothing
        if self.smoothed_saliency is None:
            self.smoothed_saliency = raw_saliency.copy()
        else:
            self.smoothed_saliency = (
                self.ema_alpha * raw_saliency + (1.0 - self.ema_alpha) * self.smoothed_saliency
            )

        return self.smoothed_saliency.astype(np.float32), latent.numpy()

    def get_top_joints(self, saliency, top_k=5):
        """Returns top-K contributing joints sorted by saliency score."""
        ranked_indices = np.argsort(saliency)[::-1][:top_k]
        multiple_hands = len(saliency) > len(JOINT_NAMES)
        return [
            {
                "joint_id": int(idx),
                "joint_name": (
                    f"{'Left' if idx < len(JOINT_NAMES) else 'Right'} "
                    f"{JOINT_NAMES[idx % len(JOINT_NAMES)]}"
                    if multiple_hands else JOINT_NAMES[idx]
                ),
                "score": float(saliency[idx])
            }
            for idx in ranked_indices
        ]

    def overlay_saliency(self, frame, raw_coords, saliency, label, confidence, latency_ms=0.0):
        """
        Renders sleek visual overlays:
        - Directional thermal bone connections
        - Heat halo rings with glowing core nodes
        - Pulsing target lock on top decision hotspot
        - Targeting bounding brackets
        - Glassmorphism HUD status telemetry
        """
        if frame is None:
            return frame

        annotated = frame.copy()
        h, w = frame.shape[:2]
        self.pulse_phase = (self.pulse_phase + 0.2) % (2 * np.pi)

        hand_count = len(raw_coords) // len(JOINT_NAMES) if raw_coords else 0
        if raw_coords is not None and hand_count and len(raw_coords) % len(JOINT_NAMES) == 0:
            overlay = frame.copy()
            glow_layer = np.zeros_like(frame)

            # Find bounding box for sci-fi bracket overlay
            xs = [p[0] for p in raw_coords]
            ys = [p[1] for p in raw_coords]
            min_x, max_x = max(0, min(xs) - 25), min(w - 1, max(xs) + 25)
            min_y, max_y = max(0, min(ys) - 25), min(h - 1, max(ys) + 25)

            # Draw sci-fi corner brackets around the hand
            self._draw_corner_brackets(overlay, min_x, min_y, max_x, max_y, (56, 189, 248))

            # Find top-1 joint for pulsating highlight
            top1_idx = int(np.argmax(saliency)) if saliency is not None else 0

            for hand_index in range(hand_count):
                offset = hand_index * len(JOINT_NAMES)
                hand_coords = raw_coords[offset:offset + len(JOINT_NAMES)]
                hand_saliency = (
                    saliency[offset:offset + len(JOINT_NAMES)]
                    if saliency is not None and len(saliency) >= offset + len(JOINT_NAMES)
                    else np.full(len(JOINT_NAMES), 0.5, dtype=np.float32)
                )
                for pt1_idx, pt2_idx in HAND_CONNECTIONS:
                    p1 = hand_coords[pt1_idx]
                    p2 = hand_coords[pt2_idx]
                    avg_heat = float((hand_saliency[pt1_idx] + hand_saliency[pt2_idx]) / 2.0)
                    bone_color = self._heat_to_color(avg_heat)
                    cv2.line(glow_layer, p1, p2, bone_color, 4, cv2.LINE_AA)
                    cv2.line(overlay, p1, p2, (240, 245, 255), 1, cv2.LINE_AA)

                for joint_index, (px, py) in enumerate(hand_coords):
                    heat = float(hand_saliency[joint_index])
                    color = self._heat_to_color(heat)
                    radius = int(4 + heat * 8)
                    cv2.circle(glow_layer, (px, py), radius + 6, color, -1, cv2.LINE_AA)
                    cv2.circle(overlay, (px, py), radius, color, -1, cv2.LINE_AA)
                    cv2.circle(overlay, (px, py), max(2, radius - 3), (255, 255, 255), -1, cv2.LINE_AA)

            # 3. Pulsing targeting ring on the primary decision joint
            top1_idx = min(top1_idx, len(raw_coords) - 1)
            top_pt = raw_coords[top1_idx]
            pulse_radius = int(14 + 5 * np.sin(self.pulse_phase))
            cv2.circle(overlay, top_pt, pulse_radius, (0, 70, 255), 2, cv2.LINE_AA)
            joint_label = JOINT_NAMES[top1_idx % len(JOINT_NAMES)]
            if hand_count > 1:
                joint_label = f"{'LEFT' if top1_idx < len(JOINT_NAMES) else 'RIGHT'} {joint_label}"
            cv2.putText(overlay, f"MAX XAI: {joint_label}",
                        (top_pt[0] + 12, top_pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1, cv2.LINE_AA)

            # Blend glow layer & overlay onto original frame
            cv2.addWeighted(glow_layer, 0.35, overlay, 0.95, 0, overlay)
            cv2.addWeighted(overlay, 0.88, annotated, 0.12, 0, annotated)

        # 4. Render Modern HUD Telemetry Card
        self._draw_modern_hud(annotated, label, confidence, saliency, latency_ms)

        return annotated

    def _heat_to_color(self, heat):
        """Converts normalized [0, 1] heat into vibrant BGR color (Blue -> Green -> Amber -> Red)."""
        u8 = np.uint8(np.clip(heat * 255, 0, 255))
        return tuple(int(c) for c in cv2.applyColorMap(np.array([[u8]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0])

    def _draw_corner_brackets(self, img, x1, y1, x2, y2, color, length=18, thickness=2):
        """Draws camera viewfinder corner brackets."""
        # Top-left
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        # Top-right
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        # Bottom-left
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        # Bottom-right
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def _draw_modern_hud(self, frame, label, confidence, saliency, latency_ms):
        """Renders frosted-glass style HUD telemetry banner."""
        h, w = frame.shape[:2]
        hud_w = min(440, w - 24)
        hud_h = 115
        x1, y1 = 14, 14
        x2, y2 = x1 + hud_w, y1 + hud_h

        # Semi-transparent dark frosted banner
        sub = frame[y1:y2, x1:x2]
        glass = np.zeros_like(sub)
        glass[:] = (15, 23, 42)  # Obsidian navy
        cv2.addWeighted(sub, 0.20, glass, 0.80, 0, frame[y1:y2, x1:x2])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (56, 189, 248), 1, cv2.LINE_AA)

        # Predicted Token & Confidence
        conf_val = float(confidence) if confidence is not None else 0.0
        label_text = f"PREDICTED SIGN: {label or '...'}"
        cv2.putText(frame, label_text, (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # Confidence percentage badge
        conf_color = (74, 222, 128) if conf_val >= 0.85 else ((56, 189, 248) if conf_val >= 0.6 else (248, 113, 113))
        conf_badge = f"{conf_val * 100:.1f}%"
        cv2.putText(frame, conf_badge, (x2 - 80, y1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.70, conf_color, 2, cv2.LINE_AA)

        # Sleek animated progress bar
        bar_x = x1 + 12
        bar_y = y1 + 40
        bar_w = hud_w - 24
        bar_h = 6
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (30, 41, 59), -1)
        fill_w = int(bar_w * np.clip(conf_val, 0.0, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), conf_color, -1)

        # Performance & Benchmark metrics
        latency_badge = "PASS" if latency_ms <= 50.0 else "WARN"
        metrics_text = f"Latency: {latency_ms:.1f}ms [{latency_badge}] | Target: <=50ms | WLASL100"
        cv2.putText(frame, metrics_text, (x1 + 12, y1 + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

        # Top 3 XAI hotspot joints
        if saliency is not None and len(saliency) == 21:
            top_joints = self.get_top_joints(saliency, top_k=3)
            hotspots_str = "XAI Hotspots: " + " | ".join([f"{j['joint_name']} ({j['score']:.2f})" for j in top_joints])
            cv2.putText(frame, hotspots_str, (x1 + 12, y1 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (251, 191, 36), 1, cv2.LINE_AA)
            cv2.putText(frame, "TCN + Transformer -> Multi-Class SVM (RBF, C=10.0)", (x1 + 12, y1 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 116, 139), 1, cv2.LINE_AA)
