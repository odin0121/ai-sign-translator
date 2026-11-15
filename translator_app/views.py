"""
Translator App Views and Streaming Endpoints
Integrates the complete pipeline: MediaPipe -> TCN -> Transformer -> SVM -> XAI -> TTS
with high-throughput Django views and REST APIs.
"""

import os
import time
import json
import base64
import logging
import re
import shutil
import tempfile
import zipfile
from string import ascii_uppercase
from functools import lru_cache
from io import BytesIO
from threading import Lock
from urllib.parse import urlencode
from collections import deque
import cv2
import numpy as np
from PIL import Image

from django.shortcuts import render
from django.http import FileResponse, JsonResponse, StreamingHttpResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.db.models import Avg, Count
from django.urls import reverse

from core_engine.landmark_extractor import (
    DUAL_HAND_FEATURE_VERSION,
    LANDMARK_FEATURE_VERSION,
    LandmarkExtractor,
    JOINT_NAMES,
)
from core_engine.hybrid_model import HybridTCNTransformer
from core_engine.svm_classifier import SVMSignClassifier, DEFAULT_VOCABULARY
from core_engine.xai_explainer import XAISaliencyExplainer
from core_engine.tts_engine import TTSEngine
from core_engine.ai_assistant import AIAssistantEngine
from .models import TranslationLog, SystemTelemetry, VocabularyItem

logger = logging.getLogger(__name__)
_learning_demo_lock = Lock()
MIN_VALID_SEQUENCE_FRAMES = 24

ISL_EMERGENCY_VOCABULARY = [
    "ACCIDENT", "CALL", "DOCTOR", "HELP", "HOT", "LOSE", "PAIN", "THIEF"
]
ASL_ALPHABET_VOCABULARY = list(ascii_uppercase)
MODEL_PROFILES = {
    "wlasl100": {
        "training_source": "wlasl100",
        "backbone_file": "tcn_transformer.weights.h5",
        "classifier_file": "svm_model.joblib",
        "vocabulary": DEFAULT_VOCABULARY,
        "name": "ASL (WLASL100)",
    },
    "isl_emergency": {
        "training_source": "isl_emergency_v1",
        "backbone_file": "isl_tcn_transformer.weights.h5",
        "classifier_file": "isl_svm_model.joblib",
        "vocabulary": ISL_EMERGENCY_VOCABULARY,
        "name": "ISL Emergency Signs",
    },
    "asl_alphabet": {
        "training_source": "asl_alphabet_v1",
        "classifier_file": "asl_alphabet_svm.joblib",
        "vocabulary": ASL_ALPHABET_VOCABULARY,
        "name": "ASL Alphabet A-Z",
    },
}


def _landmark_cache_path(model_profile):
    cache_name = {
        "isl_emergency": "isl_emergency_landmarks.npz",
        "asl_alphabet": "asl_alphabet_landmarks.npz",
    }.get(model_profile, "wlasl100_landmarks.npz")
    return os.path.join(
        settings.WEIGHTS_DIR,
        cache_name,
    )


@lru_cache(maxsize=2)
def _learning_demo_index(model_profile):
    """Map each trained label to one real training clip, never a held-out clip."""
    cache_path = _landmark_cache_path(model_profile)
    if not os.path.isfile(cache_path):
        logger.warning("Learning demos unavailable: landmark cache is missing: %s", cache_path)
        return {}

    with np.load(cache_path, allow_pickle=False) as cache:
        archive_path = str(cache["source_path"].item())
        vocabulary = [str(label) for label in cache["vocabulary"].tolist()]

        if model_profile == "isl_emergency":
            train_signers = sorted(int(signer) for signer in cache["train_signers"].tolist())
        elif model_profile == "asl_alphabet":
            demo_members = [str(member) for member in cache["demo_members"].tolist()]
        else:
            train_signers = []

    if not os.path.isfile(archive_path):
        logger.warning("Learning demos unavailable: source archive is missing: %s", archive_path)
        return {}

    demos = {}
    with zipfile.ZipFile(archive_path) as archive:
        if model_profile == "wlasl100":
            member_names = set(archive.namelist())
            class_names = {
                int(class_id): label.strip().upper()
                for class_id, label in (
                    line.split("\t", 1)
                    for line in archive.read("wlasl_class_list.txt").decode("utf-8").splitlines()
                )
            }
            metadata = json.loads(archive.read("nslt_100.json"))
            class_ids = {label: class_id for class_id, label in class_names.items()}
            for label in vocabulary:
                class_id = class_ids.get(label)
                if class_id is None:
                    continue
                for video_id, entry in sorted(metadata.items()):
                    actions = entry.get("action") or []
                    if (
                        entry.get("subset") == "train"
                        and actions
                        and int(actions[0]) == class_id
                    ):
                        member = f"videos/{str(video_id).zfill(5)}.mp4"
                        if member in member_names:
                            demos[label] = {
                                "archive_path": archive_path,
                                "member": member,
                                "format": "video/mp4",
                            }
                            break
        elif model_profile == "asl_alphabet":
            member_names = set(archive.namelist())
            for label, member in zip(vocabulary, demo_members):
                if member in member_names:
                    demos[label] = {
                        "archive_path": archive_path,
                        "member": member,
                        "format": "image/jpeg",
                    }
        else:
            cropped_archives = {
                os.path.basename(name).lower(): name
                for name in archive.namelist()
                if "/cropped_data/" in name.lower() and name.lower().endswith(".zip")
            }
            for label in vocabulary:
                inner_name = cropped_archives.get(f"{label.lower()}_cropped.zip")
                if inner_name is None:
                    continue
                for signer in train_signers:
                    member = (
                        f"{label.lower()}_Cropped/"
                        f"{label.lower()}_Crop_{signer:03d}_01.avi"
                    )
                    demos[label] = {
                        "archive_path": archive_path,
                        "archive_member": inner_name,
                        "member": member,
                        "format": "image/gif",
                    }
                    break
    return demos


def _encode_learning_gif(demo, target_path):
    """Make a small animated preview from a real AVI clip in the nested ISL archive."""
    with zipfile.ZipFile(demo["archive_path"]) as archive:
        with tempfile.NamedTemporaryFile(suffix=".zip") as nested_file:
            with archive.open(demo["archive_member"]) as source:
                shutil.copyfileobj(source, nested_file)
            nested_file.flush()

            with zipfile.ZipFile(nested_file.name) as nested_archive:
                with tempfile.NamedTemporaryFile(suffix=".avi") as clip_file:
                    clip_file.write(nested_archive.read(demo["member"]))
                    clip_file.flush()
                    capture = cv2.VideoCapture(clip_file.name)
                    if not capture.isOpened():
                        raise ValueError(f"Could not decode learning clip: {demo['member']}")
                    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = float(capture.get(cv2.CAP_PROP_FPS))
                    if frame_count < 1:
                        capture.release()
                        raise ValueError(f"Learning clip has no frames: {demo['member']}")

                    frame_indexes = np.linspace(
                        0,
                        frame_count - 1,
                        num=min(20, frame_count),
                        dtype=int,
                    )
                    frames = []
                    palette = None
                    for frame_index in frame_indexes:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                        success, frame = capture.read()
                        if not success:
                            continue
                        rgb_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        rgb_frame.thumbnail((360, 270), Image.Resampling.LANCZOS)
                        if palette is None:
                            palette = rgb_frame.convert(
                                "P",
                                palette=Image.Palette.ADAPTIVE,
                                colors=96,
                            )
                        frames.append(rgb_frame.quantize(palette=palette, dither=Image.Dither.NONE))
                    capture.release()

    if not frames:
        raise ValueError(f"Could not extract preview frames from {demo['member']}")

    clip_duration_ms = frame_count * 1000 / fps if fps > 0 else 1200
    frame_duration_ms = max(60, min(180, round(clip_duration_ms / len(frames))))
    output = BytesIO()
    frames[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=True,
    )
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=os.path.dirname(target_path),
        suffix=".gif",
        delete=False,
    ) as output_file:
        output_file.write(output.getvalue())
        temp_path = output_file.name
    os.replace(temp_path, target_path)


def learning_demo(request):
    """Serve a real training example for one selected vocabulary item."""
    if request.method != "GET":
        return JsonResponse({"error": "GET required"}, status=405)

    model_profile = request.GET.get("model_profile", "wlasl100")
    if model_profile not in MODEL_PROFILES:
        return JsonResponse({"error": "Unknown model profile"}, status=400)

    pipeline = get_pipeline(model_profile)
    token = request.GET.get("sign", "").strip().upper()
    if token not in pipeline.svm.vocabulary:
        return JsonResponse({"error": "Sign is not in this model's vocabulary"}, status=404)

    demo = _learning_demo_index(model_profile).get(token)
    if demo is None:
        return JsonResponse({"error": "No training demonstration is available for this sign"}, status=404)

    if demo["format"] == "video/mp4":
        with zipfile.ZipFile(demo["archive_path"]) as archive:
            response = HttpResponse(archive.read(demo["member"]), content_type="video/mp4")
    elif demo["format"] == "image/jpeg":
        with zipfile.ZipFile(demo["archive_path"]) as archive:
            response = HttpResponse(archive.read(demo["member"]), content_type="image/jpeg")
    else:
        archive_stat = os.stat(demo["archive_path"])
        safe_token = re.sub(r"[^A-Z0-9_-]", "_", token)
        target_path = os.path.join(
            settings.MEDIA_ROOT,
            "learning_demos",
            model_profile,
            f"{safe_token}-{archive_stat.st_mtime_ns}.gif",
        )
        with _learning_demo_lock:
            if not os.path.isfile(target_path):
                _encode_learning_gif(demo, target_path)
        response = FileResponse(open(target_path, "rb"), content_type="image/gif")

    response["Cache-Control"] = "private, max-age=86400"
    response["X-Content-Type-Options"] = "nosniff"
    return response


class SignTranslationPipeline:
    """
    Singleton service managing deep learning models, webcam frame buffers,
    and inference latency tracking.
    """
    _instances = {}
    # Saliency changes much more slowly than the prediction. Refresh its
    # gradient attribution periodically and reuse it for the intervening frames.
    XAI_REFRESH_INTERVAL = 5

    @classmethod
    def get_instance(cls, model_profile="wlasl100"):
        if model_profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown sign-language model profile: {model_profile}")
        if model_profile not in cls._instances:
            cls._instances[model_profile] = cls(model_profile)
        return cls._instances[model_profile]

    def __init__(self, model_profile="wlasl100"):
        self.model_profile = model_profile
        profile = MODEL_PROFILES[model_profile]
        logger.info("Initializing sign-translation pipeline: %s", model_profile)
        self.weights_dir = getattr(settings, 'WEIGHTS_DIR', os.path.join(settings.BASE_DIR, 'weights'))
        
        # Match training-time detector thresholds; WLASL words can use both hands.
        self.feature_version = (
            DUAL_HAND_FEATURE_VERSION
            if model_profile == "wlasl100"
            else LANDMARK_FEATURE_VERSION
        )
        self.extractor = LandmarkExtractor(
            static_image_mode=False,
            max_num_hands=2 if model_profile == "wlasl100" else 1,
            min_detection_confidence=0.3,
        )
        
        # 2. SVM vocabulary and TCN + Transformer backbone for this language profile.
        svm_weights = os.path.join(self.weights_dir, profile["classifier_file"])
        self.svm = SVMSignClassifier(
            c_param=10.0,
            kernel='rbf',
            vocabulary=profile["vocabulary"],
            model_path=svm_weights
        )
        backbone_weights = os.path.join(self.weights_dir, profile["backbone_file"])
        self.hybrid_nn = HybridTCNTransformer(
            T=60,
            D=self.extractor.feature_dim,
            latent_dim=128,
            num_classes=len(self.svm.vocabulary),
        )
        self.backbone_weights_loaded = (
            self.hybrid_nn.load_weights(backbone_weights)
            if os.path.exists(backbone_weights)
            else False
        )
        self.model_ready = bool(
            self.svm.is_fitted
            and self.svm.training_source == profile["training_source"]
            and self.svm.evaluation_metrics
            and self.svm.evaluation_metrics.get("landmark_feature_version") == self.feature_version
            and self.backbone_weights_loaded
        )
        self.test_accuracy = (
            self.svm.evaluation_metrics.get("test_accuracy")
            if self.svm.evaluation_metrics
            else None
        )
        self.min_confidence = (
            float(self.svm.evaluation_metrics.get("min_confidence", 1.0))
            if self.model_ready and self.svm.evaluation_metrics
            else 1.0
        )
        if not self.model_ready:
            logger.warning(
                "No verified model is installed for profile %s.",
                model_profile,
            )
        
        # 4. Post-Hoc XAI Explainer
        self.explainer = XAISaliencyExplainer(
            backbone=self.hybrid_nn.backbone,
            svm_classifier=self.svm
        )
        
        # 5. Non-blocking TTS Engine
        self.tts = TTSEngine(speech_rate=150)
        
        # Rolling temporal buffer for sequence window T=60
        self.window_size = 60
        self.feature_buffer = deque(maxlen=self.window_size)
        self.valid_frame_buffer = deque([False] * self.window_size, maxlen=self.window_size)
        self.valid_frame_count = 0
        
        # Initialize buffer with stationary zeros
        for _ in range(self.window_size):
            self.feature_buffer.append(np.zeros(self.extractor.feature_dim, dtype=np.float32))

        # Runtime telemetry cache
        self.total_frames = 0
        self.inference_frame_count = 0
        self.last_latency = 0.0
        self.last_prediction = "READY"
        self.last_confidence = 0.0
        self.last_saliency = np.zeros(self.extractor.feature_dim // 3, dtype=np.float32)
        self.smoothed_probs = None
        self.camera = None
        self.camera_active = False

    def process_frame(self, frame, enable_tts=False):
        """
        Executes end-to-end pipeline on a single frame:
        Landmarks -> Rolling Window -> TCN-Transformer -> SVM -> XAI -> Visualization
        """
        start_time = time.perf_counter()
        self.total_frames += 1

        # Step 1: Extract normalized landmark features and raw pixel coordinates
        normalized_pts, raw_coords = self.extractor.extract_with_raw(frame)
        detected_hands = len(raw_coords) // len(JOINT_NAMES) if raw_coords else 0
        self.feature_buffer.append(normalized_pts)
        self.valid_frame_buffer.append(bool(self.extractor.has_detection))
        self.valid_frame_count = sum(self.valid_frame_buffer)

        # Step 2: Form temporal tensor (1, 60, 63)
        seq_window = np.array(self.feature_buffer, dtype=np.float32)
        seq_tensor = np.expand_dims(seq_window, axis=0)

        # Step 3: Run the model every frame. Refresh expensive gradient-based
        # saliency every few frames and reuse the latest attribution meanwhile.
        if (
            self.extractor.has_detection
            and self.model_ready
            and self.valid_frame_count >= MIN_VALID_SEQUENCE_FRAMES
        ):
            self.inference_frame_count = getattr(self, "inference_frame_count", 0) + 1
            cached_saliency = getattr(self, "last_saliency", None)
            refresh_saliency = (
                cached_saliency is None
                or self.inference_frame_count % self.XAI_REFRESH_INTERVAL == 1
            )
            if refresh_saliency:
                saliency, latent = self.explainer.compute_saliency(seq_tensor)
                if detected_hands == 1 and len(saliency) > len(JOINT_NAMES):
                    saliency[len(JOINT_NAMES):] = 0.0
            else:
                latent = self.hybrid_nn.extract_latent(seq_tensor)
                saliency = cached_saliency
            
            # Predict with probability distribution
            if hasattr(self.svm.svm, "predict_proba"):
                curr_probs = self.svm.svm.predict_proba(latent)[0]
                if self.smoothed_probs is None or len(self.smoothed_probs) != len(curr_probs):
                    self.smoothed_probs = curr_probs.copy()
                else:
                    # Exponential Moving Average filter across frames (0.65 current, 0.35 historical)
                    self.smoothed_probs = 0.65 * curr_probs + 0.35 * self.smoothed_probs

                if hasattr(self.svm.svm, "predict"):
                    pred_idx = int(self.svm.svm.predict(latent)[0])
                else:
                    pred_idx = int(np.argmax(self.smoothed_probs))
                class_ids = getattr(self.svm.svm, "classes_", None)
                class_position = (
                    np.flatnonzero(class_ids == pred_idx)
                    if class_ids is not None
                    else np.asarray([], dtype=np.int64)
                )
                confidence = float(
                    self.smoothed_probs[class_position[0]]
                    if len(class_position)
                    else self.smoothed_probs[pred_idx]
                )
                pred_label = self.svm.vocabulary[pred_idx] if pred_idx < len(self.svm.vocabulary) else f"CLASS_{pred_idx}"

                top3_indices = np.argsort(self.smoothed_probs)[::-1][:3]
                candidates = [
                    (self.svm.vocabulary[i] if i < len(self.svm.vocabulary) else f"CLASS_{i}", float(self.smoothed_probs[i]))
                    for i in top3_indices
                ]
            else:
                pred_label, confidence, candidates = self.svm.predict(latent)

            if confidence < self.min_confidence:
                pred_label = "UNCERTAIN"
            
            # Auto-speak if confidence is robust
            if enable_tts and confidence >= self.min_confidence and confidence > 0.88:
                self.tts.speak(pred_label)

            self.last_prediction = pred_label
            self.last_confidence = confidence
            self.last_saliency = saliency
        else:
            saliency = np.zeros(self.extractor.feature_dim // 3, dtype=np.float32)
            if self.valid_frame_count < MIN_VALID_SEQUENCE_FRAMES:
                self.smoothed_probs = None
                self.inference_frame_count = 0
                self.last_saliency = None
            pred_label = (
                "MODEL NOT TRAINED"
                if self.extractor.has_detection and not self.model_ready
                else (
                    "COLLECTING SIGN"
                    if self.extractor.has_detection
                    else "WAITING FOR HAND"
                )
            )
            confidence = 0.0
            candidates = []

        self.last_prediction = pred_label
        self.last_confidence = confidence
        self.last_saliency = saliency
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_latency = elapsed_ms

        # Step 5: Render XAI Thermal Skeleton & Telemetry HUD Overlay
        annotated_frame = self.explainer.overlay_saliency(
            frame=frame,
            raw_coords=raw_coords,
            saliency=saliency,
            label=pred_label,
            confidence=confidence,
            latency_ms=elapsed_ms
        )

        top_joints = self.explainer.get_top_joints(saliency, top_k=4)

        return {
            "annotated_frame": annotated_frame,
            "prediction": pred_label,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "top_joints": top_joints,
            "candidates": candidates,
            "has_hand": self.extractor.has_detection,
            "hand_count": detected_hands,
            "model_ready": self.model_ready,
            "sequence_ready": (
                self.extractor.has_detection
                and self.valid_frame_count >= MIN_VALID_SEQUENCE_FRAMES
            ),
            "min_confidence": self.min_confidence,
            "saliency": saliency.tolist()
        }


class AlphabetSignPipeline:
    """Static A-Z classifier using the normalized hand-landmark feature pipeline."""

    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.extractor = LandmarkExtractor(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.2,
        )
        self.svm = SVMSignClassifier(
            c_param=10.0,
            kernel="rbf",
            vocabulary=ASL_ALPHABET_VOCABULARY,
            model_path=os.path.join(settings.WEIGHTS_DIR, "asl_alphabet_svm.joblib"),
        )
        metrics = self.svm.evaluation_metrics or {}
        self.model_ready = bool(
            self.svm.is_fitted
            and self.svm.training_source == "asl_alphabet_v1"
            and metrics.get("landmark_feature_version") == LANDMARK_FEATURE_VERSION
            and self.svm.vocabulary == ASL_ALPHABET_VOCABULARY
        )
        self.test_accuracy = metrics.get("test_accuracy")
        self.min_confidence = float(metrics.get("min_confidence", 0.5))
        self.tts = TTSEngine(speech_rate=150)
        self.total_frames = 0
        self.last_prediction = "READY"
        self.last_confidence = 0.0
        self.last_latency = 0.0

    def process_frame(self, frame, enable_tts=False):
        start_time = time.perf_counter()
        self.total_frames += 1
        normalized_points, raw_coords = self.extractor.extract_with_raw(frame)
        has_hand = self.extractor.has_detection
        detected_hands = len(raw_coords) // len(JOINT_NAMES) if raw_coords else 0
        candidates = []
        confidence = 0.0
        saliency = np.zeros(21, dtype=np.float32)

        if has_hand and self.model_ready:
            prediction, confidence, candidates = self.svm.predict(normalized_points)
            if confidence < self.min_confidence:
                prediction = "UNCERTAIN"
                candidates = []
            elif enable_tts and confidence > 0.88:
                self.tts.speak(prediction)
        elif has_hand:
            prediction = "MODEL NOT TRAINED"
        else:
            prediction = "WAITING FOR HAND"

        annotated_frame = frame.copy()
        if raw_coords:
            for x, y in raw_coords:
                cv2.circle(annotated_frame, (x, y), 2, (120, 240, 80), -1)
        cv2.putText(
            annotated_frame,
            f"{prediction}  {confidence * 100:.0f}%",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (120, 240, 80) if has_hand else (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        self.last_prediction = prediction
        self.last_confidence = confidence
        self.last_latency = elapsed_ms
        return {
            "annotated_frame": annotated_frame,
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "top_joints": [],
            "candidates": candidates,
            "has_hand": has_hand,
            "hand_count": detected_hands,
            "model_ready": self.model_ready,
            "sequence_ready": has_hand,
            "min_confidence": self.min_confidence,
            "saliency": saliency.tolist(),
        }


def get_pipeline(model_profile="wlasl100"):
    if model_profile == "asl_alphabet":
        return AlphabetSignPipeline.get_instance()
    return SignTranslationPipeline.get_instance(model_profile)


# --------------------------------------------------------------------------
# Page Views
# --------------------------------------------------------------------------

def index(request):
    """Main live sign translation interface."""
    pipeline = get_pipeline()
    context = {
        "vocabulary": pipeline.svm.vocabulary,
        "total_classes": len(pipeline.svm.vocabulary),
        "test_accuracy": (
            round(pipeline.test_accuracy * 100, 1)
            if pipeline.test_accuracy is not None
            else None
        ),
        "accuracy_available": pipeline.test_accuracy is not None,
        "target_latency_ms": 50.0,
        "model_ready": pipeline.model_ready,
        "model_vocabulary": len(pipeline.svm.vocabulary) if pipeline.model_ready else 0,
    }
    return render(request, 'index.html', context)


def learning_studio(request):
    """Dedicated sign lesson page, separate from the live camera studio."""
    model_profile = request.GET.get("model_profile", "wlasl100")
    if model_profile not in MODEL_PROFILES:
        model_profile = "wlasl100"
    return render(request, "learning_studio.html", {
        "initial_model_profile": model_profile,
    })


def dashboard(request):
    """Analytics and telemetry reporting dashboard."""
    pipeline = get_pipeline()
    recent_logs = TranslationLog.objects.order_by('-timestamp')[:30]
    total_translations = TranslationLog.objects.count()
    avg_latency = TranslationLog.objects.aggregate(Avg('latency_ms'))['latency_ms__avg'] or 0.0
    avg_confidence = TranslationLog.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0.0

    # Top detected signs distribution
    top_signs = list(
        TranslationLog.objects.values('predicted_token')
        .annotate(count=Count('predicted_token'))
        .order_by('-count')[:10]
    )

    context = {
        "recent_logs": recent_logs,
        "total_translations": total_translations,
        "avg_latency": round(avg_latency, 2),
        "avg_confidence": round(avg_confidence * 100, 1),
        "top_signs": json.dumps(top_signs),
        "test_accuracy": (
            round(pipeline.test_accuracy * 100, 1)
            if pipeline.test_accuracy is not None
            else None
        ),
    }
    return render(request, 'dashboard.html', context)


# --------------------------------------------------------------------------
# Streaming & REST APIs
# --------------------------------------------------------------------------

def stream_generator():
    """Generates continuous MJPEG frames for video streaming."""
    pipeline = get_pipeline()
    cap = cv2.VideoCapture(0)

    # If local hardware camera is unavailable, construct simulated interactive feed
    use_synthetic = not cap.isOpened()
    if use_synthetic:
        logger.info("Hardware camera unavailable. Streaming dynamic interactive test feed.")

    t_counter = 0
    while True:
        if not use_synthetic:
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            # Generate simulated frame with animated motion
            t_counter += 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (30, 30, 35)
            # Animated circular pulse
            cv2.circle(frame, (320, 240), int(100 + 30 * np.sin(t_counter * 0.1)), (50, 60, 80), -1)
            cv2.putText(frame, "AI SIGN TRANSLATOR: LIVE FEED", (140, 220), cv2.FONT_HERSHEY_DUPLEX, 0.7, (200, 200, 200), 1)
            cv2.putText(frame, "Webcam capture or client stream active", (150, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 150, 160), 1)

        result = pipeline.process_frame(frame, enable_tts=False)
        annotated = result["annotated_frame"]

        # Encode as JPEG
        ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.03)  # ~30 FPS throttle


def video_feed(request):
    """MJPEG streaming endpoint."""
    return StreamingHttpResponse(
        stream_generator(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@csrf_exempt
def predict_frame(request):
    """
    REST API for client-side webcam frames sent as Base64.
    Returns: JSON { token, confidence, latency_ms, top_joints, candidates, annotated_frame_b64 }
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body)
        image_data = data.get('image', '')
        enable_tts = data.get('enable_tts', False)
        model_profile = data.get('model_profile', 'wlasl100')
        if model_profile not in MODEL_PROFILES:
            return JsonResponse({'error': 'Unknown model profile'}, status=400)

        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({'error': 'Invalid image format'}, status=400)

        pipeline = get_pipeline(model_profile)
        res = pipeline.process_frame(frame, enable_tts=enable_tts)

        # Log prediction to SQLite telemetry if valid sign detected
        if (
            res["has_hand"]
            and res["model_ready"]
            and res["confidence"] >= pipeline.min_confidence
        ):
            try:
                TranslationLog.objects.create(
                    predicted_token=res["prediction"],
                    confidence=res["confidence"],
                    latency_ms=res["latency_ms"],
                    top_contributing_joints=res["top_joints"],
                    saliency_vector=res["saliency"],
                    tts_played=enable_tts,
                    source=model_profile
                )
            except Exception as e:
                logger.error(f"Error saving TranslationLog: {e}")

        # Optional: return annotated JPEG base64 for direct client canvas rendering
        _, jpeg = cv2.imencode('.jpg', res["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_b64 = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        return JsonResponse({
            'status': 'success',
            'prediction': res['prediction'],
            'confidence': round(res['confidence'], 4),
            'latency_ms': round(res['latency_ms'], 2),
            'top_joints': res['top_joints'],
            'candidates': res['candidates'],
            'has_hand': res['has_hand'],
            'hand_count': res.get('hand_count', int(bool(res['has_hand']))),
            'model_ready': res['model_ready'],
            'min_confidence': res['min_confidence'],
            'model_profile': model_profile,
            'annotated_image': f"data:image/jpeg;base64,{annotated_b64}"
        })

    except Exception as e:
        logger.exception("Error in predict_frame endpoint")
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def tts_speak(request):
    """REST API triggering speech playback for translated text."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body)
        text = data.get('text', '').strip()
        if not text:
            return JsonResponse({'error': 'Text is empty'}, status=400)

        pipeline = get_pipeline()
        success = pipeline.tts.speak(text, allow_repeat=True)
        return JsonResponse({'status': 'success', 'spoken': text, 'queued': success})

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def telemetry_stats(request):
    """Returns runtime telemetry for live HUD charts."""
    pipeline = get_pipeline()
    total_logs = TranslationLog.objects.count()
    avg_latency = TranslationLog.objects.aggregate(Avg('latency_ms'))['latency_ms__avg'] or pipeline.last_latency

    return JsonResponse({
        'fps': round(1000.0 / max(pipeline.last_latency, 1.0), 1),
        'last_latency_ms': round(pipeline.last_latency, 1),
        'avg_latency_ms': round(avg_latency, 1),
        'total_translations': total_logs,
        'last_prediction': pipeline.last_prediction,
        'last_confidence': round(pipeline.last_confidence * 100, 1),
        'test_accuracy': pipeline.test_accuracy,
        'test_accuracy_percent': (
            round(pipeline.test_accuracy * 100, 1)
            if pipeline.test_accuracy is not None
            else None
        ),
        'target_latency_ms': 50.0
    })


@csrf_exempt
def clear_logs(request):
    """Clears translation history."""
    if request.method == 'POST':
        TranslationLog.objects.all().delete()
        return JsonResponse({'status': 'success', 'message': 'All logs cleared.'})
    return JsonResponse({'error': 'POST required'}, status=405)


def vocabulary_api(request):
    """Returns vocabulary and measured model status for a selected profile."""
    model_profile = request.GET.get('model_profile', 'wlasl100')
    if model_profile not in MODEL_PROFILES:
        return JsonResponse({'error': 'Unknown model profile'}, status=400)
    pipeline = get_pipeline(model_profile)
    vocab = []
    demos = _learning_demo_index(model_profile)
    items = {v.token: v for v in VocabularyItem.objects.all()}
    for token in pipeline.svm.vocabulary:
        item = items.get(token)
        cat = item.category if item else {
            "isl_emergency": "ISL emergency sign",
            "asl_alphabet": "ASL alphabet letter",
        }.get(model_profile, "WLASL100 sign")
        desc = item.description if item else "Watch the recorded training example, then practice the sign."
        demo = demos.get(token)
        vocab.append({
            "token": token,
            "category": cat,
            "description": desc,
            "demo_url": (
                f"{reverse('learning_demo')}?{urlencode({'model_profile': model_profile, 'sign': token})}"
                if demo else None
            ),
            "demo_format": demo["format"] if demo else None,
            "test_accuracy_percent": (
                round(pipeline.test_accuracy * 100, 1)
                if pipeline.test_accuracy is not None
                else None
            )
        })
    return JsonResponse({
        "vocabulary": vocab,
        "count": len(vocab),
        "model_profile": model_profile,
        "model_name": MODEL_PROFILES[model_profile]["name"],
        "model_ready": pipeline.model_ready,
        "test_accuracy_percent": (
            round(pipeline.test_accuracy * 100, 1)
            if pipeline.test_accuracy is not None
            else None
        ),
    })


def export_csv(request):
    """Exports translation logs to a downloadable CSV file."""
    import csv
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="ai_sign_translator_telemetry.csv"'

    writer = csv.writer(response)
    writer.writerow(['Timestamp', 'Predicted Token', 'Confidence', 'Latency (ms)', 'Top XAI Joints', 'TTS Played', 'Source'])

    for log in TranslationLog.objects.all().order_by('-timestamp'):
        top_str = "; ".join([f"{j.get('joint_name')}:{j.get('score', 0):.2f}" for j in log.top_contributing_joints]) if log.top_contributing_joints else "N/A"
        writer.writerow([
            log.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            log.predicted_token,
            f"{log.confidence:.4f}",
            f"{log.latency_ms:.2f}",
            top_str,
            log.tts_played,
            log.source
        ])

    return response


def export_json(request):
    """Exports translation logs as a structured JSON file."""
    logs = []
    for log in TranslationLog.objects.all().order_by('-timestamp'):
        logs.append({
            "timestamp": log.timestamp.isoformat(),
            "predicted_token": log.predicted_token,
            "confidence": log.confidence,
            "latency_ms": log.latency_ms,
            "top_contributing_joints": log.top_contributing_joints,
            "tts_played": log.tts_played,
            "source": log.source
        })
    return JsonResponse({"telemetry_logs": logs, "count": len(logs)}, json_dumps_params={'indent': 2})


# --------------------------------------------------------------------------
# AI Assistant & Multilingual Translation APIs
# --------------------------------------------------------------------------

def ai_assistant_chat(request):
    """
    Site-help chat endpoint. Answers are generated by the local help engine.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JsonResponse({'error': 'Request body must contain valid JSON.'}, status=400)

    if not isinstance(data, dict):
        return JsonResponse({'error': 'Request body must be a JSON object.'}, status=400)

    query = data.get('query')
    if not isinstance(query, str) or not query.strip():
        return JsonResponse({'error': 'Enter a question for the assistant.'}, status=400)
    query = query.strip()
    if len(query) > 1000:
        return JsonResponse({'error': 'Questions must be 1000 characters or fewer.'}, status=400)

    model_profile = data.get('model_profile', 'wlasl100')
    if not isinstance(model_profile, str) or model_profile not in MODEL_PROFILES:
        return JsonResponse({'error': 'Choose a supported sign-language model.'}, status=400)

    pipeline = get_pipeline(model_profile)
    answer = AIAssistantEngine.answer_query(
        query,
        model_profile=model_profile,
        vocabulary=pipeline.svm.vocabulary,
    )
    return JsonResponse({
        'status': 'success',
        'query': query,
        'response': answer,
    })


@csrf_exempt
def ai_translate_sentence(request):
    """
    REST API to polish raw sign sequences and translate into Spanish, French, German, or Hindi.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        data = json.loads(request.body)
        raw_text = data.get('text', '').strip()
        target_lang = data.get('target_language', 'Spanish').strip()

        polished = AIAssistantEngine.polish_sentence(raw_text)
        translated = AIAssistantEngine.translate_sentence(polished, target_language=target_lang)

        return JsonResponse({
            'status': 'success',
            'original': raw_text,
            'polished_sentence': polished,
            'target_language': target_lang,
            'translated_sentence': translated
        })
    except Exception as e:
        logger.exception("Error in ai_translate_sentence")
        return JsonResponse({'error': str(e)}, status=500)
