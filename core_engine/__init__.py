"""
Core Engine for AI Sign Language Translator
Multi-Layered Hybrid Framework with Localized Feature Modeling & Post-Hoc XAI
"""

import sys
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('KERAS_HOME', '/tmp/keras')

# Auto-resolve workspace venv and user site-packages for cv2, mediapipe, etc.
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_workspace_root = os.path.dirname(_parent_dir)

_candidate_paths = [
    _parent_dir,
    _workspace_root,
    os.path.join(_workspace_root, 'venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
    os.path.join(_parent_dir, 'venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages'),
    os.path.expanduser(f'~/Library/Python/{sys.version_info.major}.{sys.version_info.minor}/lib/python/site-packages'),
]
for _p in _candidate_paths:
    if os.path.exists(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from .landmark_extractor import LandmarkExtractor, HAND_CONNECTIONS, JOINT_NAMES
from .hybrid_model import build_backbone, HybridTCNTransformer, HybridTranslator
from .svm_classifier import SVMSignClassifier
from .xai_explainer import XAISaliencyExplainer
from .data_loader import SignDataLoader
from .tts_engine import TTSEngine
from .ai_assistant import AIAssistantEngine

__all__ = [
    "LandmarkExtractor",
    "HAND_CONNECTIONS",
    "JOINT_NAMES",
    "build_backbone",
    "HybridTCNTransformer",
    "HybridTranslator",
    "SVMSignClassifier",
    "XAISaliencyExplainer",
    "SignDataLoader",
    "TTSEngine",
    "AIAssistantEngine",
]


