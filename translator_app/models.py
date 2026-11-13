"""
Translation Logs and Telemetry Database Models
Stores inference transactions, post-hoc XAI attributions, and runtime performance metrics.
"""

from django.db import models


class TranslationLog(models.Model):
    """
    Persists real-time sign translations with prediction confidence,
    inference latency, and XAI joint attributions.
    """
    predicted_token = models.CharField(max_length=64, db_index=True)
    confidence = models.FloatField(help_text="Softmax / Platt probability in [0, 1]")
    latency_ms = models.FloatField(help_text="End-to-end inference latency in milliseconds")
    top_contributing_joints = models.JSONField(
        default=list,
        blank=True,
        help_text="Top XAI saliency joints [{joint_name, score}, ...]"
    )
    saliency_vector = models.JSONField(
        default=list,
        blank=True,
        help_text="Complete 21-element normalized saliency array"
    )
    tts_played = models.BooleanField(default=False)
    source = models.CharField(max_length=32, default="webcam")
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Translation Log"
        verbose_name_plural = "Translation Logs"

    def __str__(self):
        return f"[{self.timestamp.strftime('%H:%M:%S')}] {self.predicted_token} ({self.confidence * 100:.1f}%, {self.latency_ms:.1f}ms)"


class SystemTelemetry(models.Model):
    """
    Tracks runtime throughput, latency benchmarks, and device performance metrics.
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    fps = models.FloatField(default=0.0)
    inference_latency_ms = models.FloatField(default=0.0)
    xai_latency_ms = models.FloatField(default=0.0)
    total_frames_processed = models.BigIntegerField(default=0)
    active_connections = models.IntegerField(default=1)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = "System Telemetry"
        verbose_name_plural = "System Telemetry Records"

    def __str__(self):
        return f"Telemetry at {self.timestamp}: {self.fps:.1f} FPS, {self.inference_latency_ms:.1f}ms inference"


class VocabularyItem(models.Model):
    """
    Dictionary of supported sign tokens, grammatical category, and descriptions.
    """
    token = models.CharField(max_length=32, unique=True)
    category = models.CharField(max_length=32, default="Alphabet")
    description = models.TextField(blank=True, default="")
    benchmark_accuracy = models.FloatField(default=99.2)

    class Meta:
        ordering = ['token']
        verbose_name = "Vocabulary Item"
        verbose_name_plural = "Vocabulary Items"

    def __str__(self):
        return f"{self.token} ({self.category})"

