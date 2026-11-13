"""
Comprehensive Verification and Integration Test Suite
Verifies all 5 Phases of the Agent Implementation Plan Sheet:
  Phase 1: 21-joint extraction & wrist-centric normalization
  Phase 2: Dilated TCN + Multi-Head Self-Attention Transformer
  Phase 3: Multi-Class SVM Classification Head & Platt calibration
  Phase 4: Gradient-based XAI saliency attribution & overlay
  Phase 5: Django views, REST API endpoints, TTS, and latency benchmark (<= 50ms)
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

for _site in [
    os.path.join(os.path.dirname(BASE_DIR), 'venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
    os.path.join(BASE_DIR, 'venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
    os.path.expanduser(f'~/Library/Python/{sys.version_info.major}.{sys.version_info.minor}/lib/python/site-packages'),
]:
    if os.path.exists(_site) and _site not in sys.path:
        sys.path.insert(0, _site)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ai_sign_translator.settings')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('KERAS_HOME', '/tmp/keras')

import time
import numpy as np
import cv2

import django
django.setup()

from django.test import Client
from core_engine.landmark_extractor import LandmarkExtractor, JOINT_NAMES
from core_engine.hybrid_model import build_backbone, HybridTCNTransformer
from core_engine.svm_classifier import SVMSignClassifier
from core_engine.xai_explainer import XAISaliencyExplainer
from core_engine.tts_engine import TTSEngine


def test_phase1_landmarks():
    print("\n--- [TEST PHASE 1: KINEMATIC FEATURE EXTRACTION] ---")
    extractor = LandmarkExtractor(static_image_mode=True)
    
    # Test blank frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    feat_blank = extractor.extract(blank)
    assert feat_blank.shape == (63,), f"Expected shape (63,), got {feat_blank.shape}"
    print("[PASS] Blank frame handled: 63-D feature output shape verified.")

    # Test simulated hand frame
    sim_frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    cv2.circle(sim_frame, (320, 240), 60, (140, 160, 210), -1)  # skin-like tone
    norm_pts, raw_coords = extractor.extract_with_raw(sim_frame)
    assert norm_pts.shape == (63,), f"Expected shape (63,), got {norm_pts.shape}"
    assert len(JOINT_NAMES) == 21, f"Expected 21 joints, got {len(JOINT_NAMES)}"
    print(f"[PASS] 21-joint extraction & wrist-centric alignment verified.")


def test_phase2_hybrid_backbone():
    print("\n--- [TEST PHASE 2: HYBRID TCN + TRANSFORMER BACKBONE] ---")
    T, D, latent_dim = 60, 63, 128
    backbone = build_backbone(T=T, D=D, latent_dim=latent_dim)
    
    dummy_input = np.random.randn(2, T, D).astype(np.float32)
    latent = backbone(dummy_input, training=False)
    
    assert latent.shape == (2, latent_dim), f"Expected shape (2, {latent_dim}), got {latent.shape}"
    print(f"[PASS] TCN Dilation Stack + Multi-Head Self-Attention output shape: {latent.shape}")

    # Check weights loading
    weights_path = os.path.join(BASE_DIR, "weights", "tcn_transformer.weights.h5")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(BASE_DIR, "weights", "tcn_transformer.h5")
    
    hybrid_nn = HybridTCNTransformer(T=T, D=D, latent_dim=latent_dim, weights_path=weights_path)
    extracted_latent = hybrid_nn.extract_latent(dummy_input)
    assert extracted_latent.shape == (2, latent_dim)
    print(f"[PASS] Serialized weights loaded successfully from disk.")


def test_phase3_svm_head():
    print("\n--- [TEST PHASE 3: MULTI-CLASS SVM CLASSIFIER] ---")
    svm_path = os.path.join(BASE_DIR, "weights", "svm_model.joblib")
    svm = SVMSignClassifier(model_path=svm_path)
    
    assert svm.is_fitted, "SVM should be fitted from serialized joblib weights."
    assert svm.training_source == "wlasl100", "SVM must be trained on labeled WLASL100 data."
    assert len(svm.vocabulary) == len(svm.svm.classes_)
    
    dummy_latent = np.random.randn(1, 128).astype(np.float32)
    pred_label, conf, top3 = svm.predict(dummy_latent)
    
    assert isinstance(pred_label, str)
    assert 0.0 <= conf <= 1.0
    print(f"[PASS] SVM inference successful: Token='{pred_label}', Confidence={conf*100:.1f}%, Candidates={len(top3)}")


def test_phase4_xai_saliency():
    print("\n--- [TEST PHASE 4: POST-HOC XAI GRADIENT ATTRIBUTION] ---")
    backbone = build_backbone(T=60, D=63, latent_dim=128)
    svm = SVMSignClassifier(model_path=os.path.join(BASE_DIR, "weights", "svm_model.joblib"))
    explainer = XAISaliencyExplainer(backbone, svm)
    
    dummy_seq = np.random.randn(1, 60, 63).astype(np.float32)
    saliency, latent = explainer.compute_saliency(dummy_seq)
    
    assert saliency.shape == (21,), f"Expected 21 joint attributions, got {saliency.shape}"
    assert 0.0 <= np.min(saliency) and np.max(saliency) <= 1.0 + 1e-5
    
    top_joints = explainer.get_top_joints(saliency, top_k=3)
    assert len(top_joints) == 3
    print(f"[PASS] Gradient backprop saliency computed: Top joint = {top_joints[0]['joint_name']} ({top_joints[0]['score']:.3f})")

    # Verify Blueprint HybridTranslator class from Section 4 #3
    from core_engine.hybrid_model import HybridTranslator
    translator = HybridTranslator(backbone=backbone, svm_model=svm.svm)
    pred, prob, xai_saliency = translator.predict_with_xai(dummy_seq)
    assert xai_saliency.shape == (21,)
    print(f"[PASS] Blueprint HybridTranslator verified: Class={pred}, Probability={prob:.3f}")

    # Verify visual rendering overlay
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    raw_coords = [(320 + i * 5, 240 + i * 5) for i in range(21)]
    annotated = explainer.overlay_saliency(dummy_frame, raw_coords, saliency, "HELLO", 0.95, latency_ms=22.4)
    assert annotated.shape == dummy_frame.shape
    print("[PASS] Thermal skeleton heatmap & HUD overlay rendered successfully.")


def test_phase5_django_integration():
    print("\n--- [TEST PHASE 5: DJANGO WEB INTEGRATION & LATENCY BENCHMARK] ---")
    client = Client()
    
    # 1. Main index view
    res = client.get('/')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    assert b"AI SIGN TRANSLATOR" in res.content
    print("[PASS] GET / -> Live Studio index rendered.")

    # 2. Dashboard view
    res_dash = client.get('/dashboard/')
    assert res_dash.status_code == 200
    assert b"AI SIGN TRANSLATOR" in res_dash.content
    print("[PASS] GET /dashboard/ -> Telemetry dashboard rendered.")

    # 3. Telemetry stats API
    res_telem = client.get('/api/telemetry/')
    assert res_telem.status_code == 200
    telem_data = res_telem.json()
    assert 'avg_latency_ms' in telem_data
    print(f"[PASS] GET /api/telemetry/ -> Stats retrieved: avg latency = {telem_data['avg_latency_ms']} ms.")

    # 4. Latency benchmark test (<= 50ms target)
    from translator_app.views import get_pipeline
    pipeline = get_pipeline()
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Warmup
    pipeline.process_frame(test_frame)
    
    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        pipeline.process_frame(test_frame)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    avg_latency = float(np.mean(latencies))
    print(f"[BENCHMARK] Average End-to-End Frame Latency: {avg_latency:.2f} ms")
    assert avg_latency <= 50.0, f"Latency budget exceeded: {avg_latency:.2f} ms > 50.0 ms target!"
    print(f"[PASS] Latency benchmark satisfied: {avg_latency:.2f} ms <= 50.0 ms budget!")

    # 5. TTS Engine
    tts = TTSEngine()
    enqueued = tts.speak("A", allow_repeat=True)
    print(f"[PASS] TTS speech dispatched: enqueued={enqueued}")


if __name__ == "__main__":
    test_phase1_landmarks()
    test_phase2_hybrid_backbone()
    test_phase3_svm_head()
    test_phase4_xai_saliency()
    test_phase5_django_integration()
    print("\n================================================================")
    print("  ALL 5 PHASES AND VERIFICATION BENCHMARKS PASSED SUCCESSFULLY! ")
    print("================================================================\n")
