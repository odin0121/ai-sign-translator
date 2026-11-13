"""
Django Unit and Integration Tests for translator_app
"""

from collections import deque
import io
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.test import TestCase, Client
from django.urls import reverse
from .models import TranslationLog, SystemTelemetry, VocabularyItem
from core_engine.landmark_extractor import LandmarkExtractor
from core_engine.data_loader import SignDataLoader
from core_engine.svm_classifier import SVMSignClassifier
from .views import ASL_ALPHABET_VOCABULARY, ISL_EMERGENCY_VOCABULARY, SignTranslationPipeline
import json
import base64
import numpy as np
import cv2
import zipfile


class TranslatorAppTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.vocab_item = VocabularyItem.objects.create(
            token="HELLO",
            category="Expression",
            description="ASL greeting sign",
            benchmark_accuracy=99.2
        )

    def test_index_page(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI SIGN TRANSLATOR")
        self.assertContains(response, "Camera &amp; Vision Studio")
        self.assertContains(response, 'id="model-profile-select"')
        self.assertContains(response, "ISL — Emergency signs")
        self.assertContains(response, "ASL Alphabet · A–Z")
        self.assertContains(response, "Learn signs")
        self.assertContains(response, 'id="assistant-chat-toggle"')
        self.assertContains(response, "Built-in site help")

    def test_learning_studio_is_a_separate_page(self):
        response = self.client.get(
            reverse('learning_studio'),
            {'model_profile': 'asl_alphabet'},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "LEARN AT YOUR OWN PACE")
        self.assertContains(response, 'id="learning-sign-list"')
        self.assertContains(response, 'id="assistant-chat-toggle"')
        self.assertContains(response, 'value="asl_alphabet" selected')
        self.assertContains(response, 'data-model-profile="wlasl100"')
        self.assertContains(response, 'data-model-profile="asl_alphabet"')
        self.assertContains(response, 'data-model-profile="isl_emergency"')
        self.assertContains(response, "Indian Sign Language")
        self.assertNotContains(response, 'class="panel camera-panel"')

    def test_dashboard_page(self):
        # Create a sample log
        TranslationLog.objects.create(
            predicted_token="HELLO",
            confidence=0.98,
            latency_ms=21.5,
            top_contributing_joints=[{"joint_name": "Thumb_TIP", "score": 0.95}],
            tts_played=True
        )
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI SIGN TRANSLATOR")
        self.assertContains(response, "HELLO")
        self.assertContains(response, 'id="assistant-chat-toggle"')

    def test_telemetry_api(self):
        response = self.client.get(reverse('telemetry_stats'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('avg_latency_ms', data)
        self.assertIn('test_accuracy_percent', data)
        if data['test_accuracy_percent'] is not None:
            self.assertGreaterEqual(data['test_accuracy_percent'], 0)
            self.assertLessEqual(data['test_accuracy_percent'], 100)

    def test_vocabulary_api_returns_selected_isl_profile(self):
        pipeline = SimpleNamespace(
            svm=SimpleNamespace(vocabulary=ISL_EMERGENCY_VOCABULARY),
            test_accuracy=0.75,
            model_ready=True,
        )
        with (
            patch('translator_app.views.get_pipeline', return_value=pipeline) as get_model,
            patch(
                'translator_app.views._learning_demo_index',
                return_value={
                    token: {"format": "image/gif"}
                    for token in ISL_EMERGENCY_VOCABULARY
                },
            ),
        ):
            response = self.client.get(
                reverse('vocabulary_api'),
                {'model_profile': 'isl_emergency'},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['model_profile'], 'isl_emergency')
        self.assertEqual(data['count'], 8)
        self.assertTrue(data['model_ready'])
        self.assertEqual(data['test_accuracy_percent'], 75.0)
        self.assertEqual(
            [item['token'] for item in data['vocabulary']],
            ISL_EMERGENCY_VOCABULARY,
        )
        self.assertTrue(all(item['demo_url'] for item in data['vocabulary']))
        self.assertTrue(all(item['demo_format'] == 'image/gif' for item in data['vocabulary']))
        get_model.assert_called_once_with('isl_emergency')

    def test_vocabulary_api_returns_all_alphabet_letters(self):
        pipeline = SimpleNamespace(
            svm=SimpleNamespace(vocabulary=ASL_ALPHABET_VOCABULARY),
            test_accuracy=0.8,
            model_ready=True,
        )
        with (
            patch('translator_app.views.get_pipeline', return_value=pipeline),
            patch(
                'translator_app.views._learning_demo_index',
                return_value={
                    token: {"format": "image/jpeg"}
                    for token in ASL_ALPHABET_VOCABULARY
                },
            ),
        ):
            response = self.client.get(
                reverse('vocabulary_api'),
                {'model_profile': 'asl_alphabet'},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['count'], 26)
        self.assertEqual(
            [item['token'] for item in data['vocabulary']],
            list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
        )
        self.assertTrue(all(item['demo_url'] for item in data['vocabulary']))
        self.assertTrue(all(item['demo_format'] == 'image/jpeg' for item in data['vocabulary']))
        self.assertTrue(all(item['category'] == 'ASL alphabet letter' for item in data['vocabulary']))

    def test_learning_demo_serves_only_a_selected_profile_training_clip(self):
        pipeline = SimpleNamespace(svm=SimpleNamespace(vocabulary=['HELLO']))
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, 'training.zip')
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('videos/00001.mp4', b'test mp4 bytes')

            demo_index = {
                'HELLO': {
                    'archive_path': archive_path,
                    'member': 'videos/00001.mp4',
                    'format': 'video/mp4',
                }
            }
            with (
                patch('translator_app.views.get_pipeline', return_value=pipeline),
                patch('translator_app.views._learning_demo_index', return_value=demo_index),
            ):
                response = self.client.get(
                    reverse('learning_demo'),
                    {'model_profile': 'wlasl100', 'sign': 'hello'},
                )
                missing_sign = self.client.get(
                    reverse('learning_demo'),
                    {'model_profile': 'wlasl100', 'sign': 'UNKNOWN'},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'video/mp4')
        self.assertEqual(response.content, b'test mp4 bytes')
        self.assertEqual(response['X-Content-Type-Options'], 'nosniff')
        self.assertEqual(missing_sign.status_code, 404)

    def test_alphabet_learning_demo_is_served_as_jpeg(self):
        pipeline = SimpleNamespace(svm=SimpleNamespace(vocabulary=['A']))
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, 'alphabet.zip')
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('Gesture Image Data/A/a.jpg', b'jpeg sample')

            with (
                patch('translator_app.views.get_pipeline', return_value=pipeline),
                patch(
                    'translator_app.views._learning_demo_index',
                    return_value={
                        'A': {
                            'archive_path': archive_path,
                            'member': 'Gesture Image Data/A/a.jpg',
                            'format': 'image/jpeg',
                        }
                    },
                ),
            ):
                response = self.client.get(
                    reverse('learning_demo'),
                    {'model_profile': 'asl_alphabet', 'sign': 'a'},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'image/jpeg')
        self.assertEqual(response.content, b'jpeg sample')
        self.assertEqual(response['X-Content-Type-Options'], 'nosniff')

    def test_predict_api_valid_frame(self):
        # Generate dummy JPEG
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', dummy_img)
        b64_str = base64.b64encode(jpeg.tobytes()).decode('utf-8')

        payload = {
            'image': f"data:image/jpeg;base64,{b64_str}",
            'enable_tts': False
        }
        response = self.client.post(
            reverse('predict_frame'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('prediction', data)
        self.assertIn('latency_ms', data)
        self.assertFalse(data['has_hand'])
        self.assertEqual(data['prediction'], 'WAITING FOR HAND')

    def test_predict_api_uses_selected_model_profile(self):
        dummy_img = np.zeros((32, 32, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', dummy_img)
        b64_str = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        result = {
            'annotated_frame': dummy_img,
            'prediction': 'WAITING FOR HAND',
            'confidence': 0.0,
            'latency_ms': 2.0,
            'top_joints': [],
            'candidates': [],
            'has_hand': False,
            'model_ready': True,
            'min_confidence': 0.5,
            'saliency': [],
        }
        pipeline = SimpleNamespace(process_frame=Mock(return_value=result))

        with patch('translator_app.views.get_pipeline', return_value=pipeline) as get_model:
            response = self.client.post(
                reverse('predict_frame'),
                data=json.dumps({
                    'image': f'data:image/jpeg;base64,{b64_str}',
                    'model_profile': 'isl_emergency',
                }),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['model_profile'], 'isl_emergency')
        get_model.assert_called_once_with('isl_emergency')

    def test_predict_api_rejects_unknown_model_profile(self):
        response = self.client.post(
            reverse('predict_frame'),
            data=json.dumps({'image': 'unused', 'model_profile': 'unknown'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_skin_colored_non_hand_is_not_a_hand(self):
        extractor = object.__new__(LandmarkExtractor)
        extractor.neural_detector = None
        extractor.hands = None
        extractor.has_detection = False
        extractor.last_valid_pts = np.zeros(63, dtype=np.float32)

        frame = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.rectangle(frame, (270, 170), (370, 430), (85, 130, 190), -1)
        points, raw_coords = extractor.extract_with_raw(frame)

        self.assertFalse(extractor.has_detection)
        self.assertIsNone(raw_coords)
        self.assertFalse(np.any(points))

    def test_two_hand_landmark_features_keep_both_hands_in_stable_slots(self):
        extractor = object.__new__(LandmarkExtractor)
        extractor.hand_count = 2
        extractor.feature_dim = 126

        def make_hand(wrist_x):
            return [
                SimpleNamespace(
                    x=wrist_x + joint * 0.001,
                    y=0.4 + joint * 0.001,
                    z=joint * 0.0001,
                )
                for joint in range(21)
            ]

        right_hand = make_hand(0.75)
        left_hand = make_hand(0.2)
        features, raw_coords = extractor._normalize_hands(
            [right_hand, left_hand],
            width=640,
            height=480,
        )

        self.assertEqual(features.shape, (126,))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(raw_coords), 42)
        self.assertLess(raw_coords[0][0], raw_coords[21][0])
        self.assertGreater(np.linalg.norm(features[:63]), 0)
        self.assertGreater(np.linalg.norm(features[63:]), 0)

        single_hand_features, single_hand_coords = extractor._normalize_hands(
            [left_hand],
            width=640,
            height=480,
        )
        self.assertEqual(single_hand_features.shape, (126,))
        self.assertEqual(len(single_hand_coords), 21)
        self.assertGreater(np.linalg.norm(single_hand_features[:63]), 0)
        self.assertEqual(np.linalg.norm(single_hand_features[63:]), 0)

    def test_predict_api_does_not_accept_skin_blob_as_hand(self):
        image = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (270, 170), (370, 430), (85, 130, 190), -1)
        _, jpeg = cv2.imencode('.jpg', image)
        image_data = base64.b64encode(jpeg.tobytes()).decode('utf-8')
        response = self.client.post(
            reverse('predict_frame'),
            data=json.dumps({'image': f'data:image/jpeg;base64,{image_data}'}),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data['has_hand'])
        self.assertEqual(data['prediction'], 'WAITING FOR HAND')

    def test_video_prediction_survives_intermittent_hand_detection(self):
        class FakeSVM:
            def predict_proba(self, _latent):
                return np.array([[0.08, 0.92]], dtype=np.float32)

        pipeline = object.__new__(SignTranslationPipeline)
        pipeline.total_frames = 0
        pipeline.extractor = SimpleNamespace(
            extract_with_raw=lambda _frame: (np.ones(63, dtype=np.float32), []),
            has_detection=True,
        )
        pipeline.feature_buffer = deque(
            [np.zeros(63, dtype=np.float32) for _ in range(59)],
            maxlen=60,
        )
        pipeline.valid_frame_buffer = deque(
            ([True, False] * 23 + [False] * 13),
            maxlen=60,
        )
        pipeline.valid_frame_count = 23
        pipeline.window_size = 60
        pipeline.model_ready = True
        pipeline.min_confidence = 0.95
        pipeline.svm = SimpleNamespace(
            svm=FakeSVM(),
            vocabulary=["NO", "YES"],
        )
        pipeline.smoothed_probs = None
        pipeline.explainer = Mock()
        pipeline.explainer.compute_saliency.return_value = (
            np.zeros(21, dtype=np.float32),
            np.zeros((1, 128), dtype=np.float32),
        )
        pipeline.explainer.overlay_saliency.side_effect = lambda **kwargs: kwargs["frame"]
        pipeline.explainer.get_top_joints.return_value = []
        pipeline.tts = Mock()

        result = pipeline.process_frame(np.zeros((32, 32, 3), dtype=np.uint8), enable_tts=True)

        self.assertEqual(result["prediction"], "UNCERTAIN")
        self.assertAlmostEqual(result["confidence"], 0.92)
        self.assertEqual(result["candidates"][0][0], "YES")
        self.assertAlmostEqual(result["candidates"][0][1], 0.92)
        self.assertEqual(pipeline.valid_frame_count, 24)
        pipeline.tts.speak.assert_not_called()

    def test_isl_emergency_loader_keeps_signers_disjoint(self):
        labels = ["accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"]
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = os.path.join(temp_dir, "isl.zip")
            cache_path = os.path.join(temp_dir, "landmarks.npz")
            with zipfile.ZipFile(archive_path, "w") as outer:
                for label in labels:
                    inner_buffer = io.BytesIO()
                    with zipfile.ZipFile(inner_buffer, "w") as inner:
                        for signer in range(1, 6):
                            for repetition in range(1, 3):
                                clip_name = (
                                    f"{label}_Cropped/{label}_Crop_{signer:03}_{repetition:02}.avi"
                                )
                                inner.writestr(clip_name, b"test video")
                    outer.writestr(
                        f"dataset/Cropped_Data/{label}_Cropped.zip",
                        inner_buffer.getvalue(),
                    )

            loader = object.__new__(SignDataLoader)
            loader.sequence_length = 60
            loader.feature_dim = 63
            loader.process_video_clip = lambda _path: np.ones((60, 63), dtype=np.float32)
            data = loader.load_isl_emergency_archive(
                archive_path,
                cache_path=cache_path,
                seed=7,
            )

            X_train, y_train, X_val, y_val, X_test, y_test, vocabulary = data
            with np.load(cache_path, allow_pickle=False) as cached:
                train_signers = set(cached["train_signers"].tolist())
                val_signers = set(cached["val_signers"].tolist())
                test_signers = set(cached["test_signers"].tolist())

            self.assertEqual(vocabulary, [label.upper() for label in labels])
            self.assertEqual(X_train.shape[1:], (60, 63))
            self.assertEqual(X_val.shape[1:], (60, 63))
            self.assertEqual(X_test.shape[1:], (60, 63))
            self.assertEqual(len(np.unique(y_train)), 8)
            self.assertEqual(len(np.unique(y_val)), 8)
            self.assertEqual(len(np.unique(y_test)), 8)
            self.assertFalse(train_signers & val_signers)
            self.assertFalse(train_signers & test_signers)
            self.assertFalse(val_signers & test_signers)

    def test_speak_api(self):
        payload = {'text': 'HELLO'}
        response = self.client.post(
            reverse('tts_speak'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['spoken'], 'HELLO')

    def test_translate_api(self):
        payload = {'text': 'HELLO THANK YOU', 'target_language': 'Spanish'}
        response = self.client.post(
            reverse('ai_translate_sentence'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('polished_sentence', data)
        self.assertIn('translated_sentence', data)

    def test_untrained_svm_does_not_invent_a_prediction(self):
        classifier = SVMSignClassifier()
        with self.assertRaisesRegex(RuntimeError, "has not been trained"):
            classifier.predict(np.zeros(128, dtype=np.float32))

    def test_svm_prediction_and_evaluation_use_the_probability_winner(self):
        class ProbabilityModel:
            classes_ = np.asarray([0, 1])

            def predict(self, features):
                return np.zeros(len(features), dtype=np.int32)

            def predict_proba(self, features):
                return np.tile(np.asarray([0.1, 0.9]), (len(features), 1))

        classifier = SVMSignClassifier(vocabulary=["A", "B"])
        classifier.svm = ProbabilityModel()
        classifier.is_fitted = True

        label, confidence, candidates = classifier.predict(np.zeros(2, dtype=np.float32))
        evaluation = classifier.evaluate(
            np.zeros((2, 2), dtype=np.float32),
            np.asarray([0, 1]),
        )

        self.assertEqual(label, "B")
        self.assertEqual(confidence, 0.9)
        self.assertEqual(candidates[0][0], "B")
        self.assertEqual(evaluation["accuracy"], 0.5)

    def test_assistant_chat_api(self):
        payload = {'query': 'How to sign L?'}
        response = self.client.post(
            reverse('ai_assistant_chat'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('response', data)
        self.assertIn('general ASL reference guidance', data['response'])

    def test_assistant_chat_answers_site_learning_questions(self):
        response = self.client.post(
            reverse('ai_assistant_chat'),
            data=json.dumps({
                'query': 'How do I use Learn signs?',
                'model_profile': 'wlasl100',
            }),
            content_type='application/json',
        )

        self.assertEqual(response.status_code, 200)
        answer = response.json()['response']
        self.assertIn('training data', answer)
        self.assertIn('Practice with camera', answer)
        self.assertIn('separately for each model', answer)

    def test_assistant_chat_uses_selected_model_vocabulary(self):
        pipeline = SimpleNamespace(
            svm=SimpleNamespace(vocabulary=['CALL', 'HELP']),
        )
        with patch('translator_app.views.get_pipeline', return_value=pipeline) as get_model:
            response = self.client.post(
                reverse('ai_assistant_chat'),
                data=json.dumps({
                    'query': 'What signs are supported?',
                    'model_profile': 'isl_emergency',
                }),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 200)
        answer = response.json()['response']
        self.assertIn('ISL Emergency', answer)
        self.assertIn('2 labels', answer)
        self.assertIn('CALL, HELP', answer)
        get_model.assert_called_once_with('isl_emergency')

    def test_assistant_chat_explains_alphabet_model(self):
        pipeline = SimpleNamespace(
            svm=SimpleNamespace(vocabulary=['A', 'B', 'C']),
        )
        with patch('translator_app.views.get_pipeline', return_value=pipeline):
            response = self.client.post(
                reverse('ai_assistant_chat'),
                data=json.dumps({
                    'query': 'What signs are supported?',
                    'model_profile': 'asl_alphabet',
                }),
                content_type='application/json',
            )

        self.assertEqual(response.status_code, 200)
        answer = response.json()['response']
        self.assertIn('ASL Alphabet · A–Z', answer)
        self.assertIn('3 labels', answer)
        self.assertIn('A, B, C', answer)

    def test_assistant_chat_rejects_invalid_requests(self):
        endpoint = reverse('ai_assistant_chat')
        empty_question = self.client.post(
            endpoint,
            data=json.dumps({'query': '  '}),
            content_type='application/json',
        )
        invalid_profile = self.client.post(
            endpoint,
            data=json.dumps({'query': 'How does the site work?', 'model_profile': 'unknown'}),
            content_type='application/json',
        )
        invalid_json = self.client.post(
            endpoint,
            data='{',
            content_type='application/json',
        )

        self.assertEqual(empty_question.status_code, 400)
        self.assertEqual(invalid_profile.status_code, 400)
        self.assertEqual(invalid_json.status_code, 400)

    def test_assistant_chat_requires_csrf_token(self):
        csrf_client = Client(enforce_csrf_checks=True)
        csrf_client.get(reverse('index'))
        token = csrf_client.cookies['csrftoken'].value
        endpoint = reverse('ai_assistant_chat')
        payload = json.dumps({'query': 'How do I get started?'})

        rejected = csrf_client.post(
            endpoint,
            data=payload,
            content_type='application/json',
        )
        accepted = csrf_client.post(
            endpoint,
            data=payload,
            content_type='application/json',
            HTTP_X_CSRFTOKEN=token,
        )

        self.assertEqual(rejected.status_code, 403)
        self.assertEqual(accepted.status_code, 200)
