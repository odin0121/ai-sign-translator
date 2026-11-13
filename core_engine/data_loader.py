"""
Data Loader & Dataset Ingestion Module
Handles WLASL dynamic video clips, ASL Alphabet static images,
and produces memory-mapped (N, 60, 63) tensors for the hybrid pipeline.
"""

import os
import glob
import json
import re
import tempfile
import zipfile
import cv2
import numpy as np
import h5py
import logging
from .landmark_extractor import (
    DUAL_HAND_FEATURE_VERSION,
    LANDMARK_FEATURE_VERSION,
    LandmarkExtractor,
    JOINT_NAMES,
)
from .svm_classifier import DEFAULT_VOCABULARY

logger = logging.getLogger(__name__)


class SignDataLoader:
    """
    Ingests and normalizes sign language datasets (WLASL dynamic videos & ASL Alphabet images)
    into tensors of shape (N, T=60, D=63), or two-hand WLASL tensors (N, T=60, D=126).
    """

    def __init__(self, sequence_length=60, feature_dim=63, vocabulary=None):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.landmark_feature_version = (
            DUAL_HAND_FEATURE_VERSION if feature_dim == 126 else LANDMARK_FEATURE_VERSION
        )
        self.vocabulary = vocabulary or list(DEFAULT_VOCABULARY)
        self.label_to_idx = {label: i for i, label in enumerate(self.vocabulary)}
        self.extractor = LandmarkExtractor(
            static_image_mode=False,
            max_num_hands=2 if feature_dim == 126 else 1,
        )

    def process_video_clip(self, video_path):
        """
        Ingests a dynamic MP4 video clip, resamples/pads to T=60 frames,
        and extracts normalized 21-joint coordinates.
        Returns:
            np.ndarray of shape (60, feature_dim)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > self.sequence_length:
                indices = np.linspace(
                    0, frame_count - 1, self.sequence_length, dtype=np.int32
                )
                for index in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
        finally:
            cap.release()

        if not frames:
            logger.warning(f"Could not read frames from {video_path}")
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        return self.extractor.extract_sequence(frames, target_T=self.sequence_length)

    def load_wlasl100_archive(self, archive_path, cache_path=None):
        """
        Load WLASL100 clips from the official archive and its provided split files.

        Returns train, validation, and test sequences and labels, plus the mapped
        vocabulary. Failed or unreadable clips are skipped rather than assigned
        fabricated zero-valued landmarks.
        """
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=False) as cached:
                stat = os.stat(archive_path) if os.path.isfile(archive_path) else None
                cache_matches = (
                    stat is not None
                    and "landmark_feature_version" in cached.files
                    and int(cached["landmark_feature_version"])
                    == getattr(self, "landmark_feature_version", LANDMARK_FEATURE_VERSION)
                    and str(os.path.realpath(archive_path)) == str(cached["source_path"].item())
                    and stat.st_size == int(cached["source_size"])
                    and stat.st_mtime_ns == int(cached["source_mtime_ns"])
                )
                if cache_matches:
                    vocabulary = cached["vocabulary"].astype(str).tolist()
                    return (
                        cached["X_train"],
                        cached["y_train"],
                        cached["X_val"],
                        cached["y_val"],
                        cached["X_test"],
                        cached["y_test"],
                        vocabulary,
                    )

        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"WLASL archive not found: {archive_path}")

        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            required_files = ("nslt_100.json", "wlasl_class_list.txt")
            missing = [name for name in required_files if name not in members]
            if missing:
                raise ValueError(f"Not a WLASL100 archive; missing {', '.join(missing)}")

            class_names = {}
            for line in archive.read("wlasl_class_list.txt").decode("utf-8").splitlines():
                class_id, label = line.split("\t", 1)
                class_names[int(class_id)] = label.strip().upper()

            split_data = json.loads(archive.read("nslt_100.json"))
            video_members = {
                os.path.splitext(os.path.basename(name))[0].zfill(5): name
                for name in members
                if name.startswith("videos/") and name.lower().endswith(".mp4")
            }

            examples = {"train": [], "val": [], "test": []}
            used_class_ids = set()
            for video_id, metadata in split_data.items():
                split = metadata.get("subset")
                action = metadata.get("action", [])
                if split not in examples or not action:
                    continue
                class_id = int(action[0])
                video_id = str(video_id).zfill(5)
                member = video_members.get(video_id)
                label = class_names.get(class_id)
                if member and label:
                    examples[split].append((member, class_id, label))
                    used_class_ids.add(class_id)

            if len(used_class_ids) < 2:
                raise ValueError("WLASL100 archive contains fewer than two labeled classes.")

            vocabulary = [class_names[index] for index in sorted(used_class_ids)]
            label_to_index = {label: index for index, label in enumerate(vocabulary)}
            arrays = {}
            labels = {}
            for split, split_examples in examples.items():
                split_sequences = []
                split_labels = []
                for position, (member, _class_id, label) in enumerate(split_examples, start=1):
                    with tempfile.NamedTemporaryFile(suffix=".mp4") as clip_file:
                        clip_file.write(archive.read(member))
                        clip_file.flush()
                        sequence = self.process_video_clip(clip_file.name)

                    if not np.isfinite(sequence).all() or not np.any(sequence):
                        logger.warning("Skipping clip with no usable landmarks: %s", member)
                        continue
                    split_sequences.append(sequence)
                    split_labels.append(label_to_index[label])
                    if position % 50 == 0:
                        logger.info("Processed %d/%d WLASL %s clips", position, len(split_examples), split)

                if not split_sequences:
                    raise ValueError(f"No usable landmark sequences found in WLASL {split} split.")
                arrays[f"X_{split}"] = np.asarray(split_sequences, dtype=np.float32)
                labels[f"y_{split}"] = np.asarray(split_labels, dtype=np.int32)
                logger.info("Usable WLASL %s clips: %d", split, len(split_sequences))

        train_class_ids = set(labels["y_train"].tolist())
        if len(train_class_ids) < 2:
            raise ValueError("At least two WLASL classes need usable training clips.")
        kept_class_ids = sorted(train_class_ids)
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(kept_class_ids)}
        vocabulary = [vocabulary[class_id] for class_id in kept_class_ids]
        for split in ("train", "val", "test"):
            keep_rows = np.isin(labels[f"y_{split}"], kept_class_ids)
            arrays[f"X_{split}"] = arrays[f"X_{split}"][keep_rows]
            labels[f"y_{split}"] = np.asarray(
                [old_to_new[int(value)] for value in labels[f"y_{split}"][keep_rows]],
                dtype=np.int32,
            )
            if not len(labels[f"y_{split}"]):
                raise ValueError(f"No WLASL {split} clips remain for trained classes.")

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            stat = os.stat(archive_path)
            np.savez(
                cache_path,
                **arrays,
                **labels,
                vocabulary=np.asarray(vocabulary, dtype=np.str_),
                landmark_feature_version=np.asarray(
                    getattr(self, "landmark_feature_version", LANDMARK_FEATURE_VERSION)
                ),
                source_path=np.asarray(os.path.realpath(archive_path)),
                source_size=np.asarray(stat.st_size),
                source_mtime_ns=np.asarray(stat.st_mtime_ns),
            )
            logger.info("Cached WLASL landmark sequences to %s", cache_path)

        return (
            arrays["X_train"],
            labels["y_train"],
            arrays["X_val"],
            labels["y_val"],
            arrays["X_test"],
            labels["y_test"],
            vocabulary,
        )

    def load_isl_emergency_archive(self, archive_path, cache_path=None, seed=42):
        """Load the cropped Indian Sign Language emergency clips using signer splits."""
        if not os.path.isfile(archive_path):
            raise FileNotFoundError(f"ISL emergency archive not found: {archive_path}")

        stat = os.stat(archive_path)
        if cache_path and os.path.exists(cache_path):
            with np.load(cache_path, allow_pickle=False) as cached:
                cache_matches = (
                    "landmark_feature_version" in cached.files
                    and int(cached["landmark_feature_version"])
                    == getattr(self, "landmark_feature_version", LANDMARK_FEATURE_VERSION)
                    and str(os.path.realpath(archive_path)) == str(cached["source_path"].item())
                    and stat.st_size == int(cached["source_size"])
                    and stat.st_mtime_ns == int(cached["source_mtime_ns"])
                    and int(cached["split_seed"]) == seed
                )
                if cache_matches:
                    vocabulary = cached["vocabulary"].astype(str).tolist()
                    return (
                        cached["X_train"],
                        cached["y_train"],
                        cached["X_val"],
                        cached["y_val"],
                        cached["X_test"],
                        cached["y_test"],
                        vocabulary,
                    )

        expected_labels = {
            "accident", "call", "doctor", "help", "hot", "lose", "pain", "thief"
        }
        examples = []
        signers = set()
        with zipfile.ZipFile(archive_path) as archive:
            crop_archives = {
                os.path.basename(name).lower(): name
                for name in archive.namelist()
                if "/cropped_data/" in name.lower() and name.lower().endswith(".zip")
            }
            for label in sorted(expected_labels):
                inner_name = crop_archives.get(f"{label}_cropped.zip")
                if inner_name is None:
                    raise ValueError(f"ISL archive is missing the cropped {label} videos.")

                with tempfile.NamedTemporaryFile(suffix=".zip") as inner_file:
                    with archive.open(inner_name) as source:
                        while chunk := source.read(1024 * 1024):
                            inner_file.write(chunk)
                    inner_file.flush()

                    with zipfile.ZipFile(inner_file.name) as clips:
                        video_names = sorted(
                            name for name in clips.namelist()
                            if name.lower().endswith((".avi", ".mp4"))
                        )
                        for member in video_names:
                            match = re.search(
                                r"_(\d{3})_(\d+)\.(?:avi|mp4)$",
                                os.path.basename(member),
                                re.IGNORECASE,
                            )
                            if not match:
                                logger.warning("Skipping ISL clip with unknown signer: %s", member)
                                continue
                            signer = int(match.group(1))
                            signers.add(signer)
                            examples.append((label.upper(), signer, inner_name, member))

        if not signers:
            raise ValueError("No signer-labeled cropped ISL videos were found.")

        rng = np.random.default_rng(seed)
        signer_order = rng.permutation(sorted(signers)).tolist()
        test_count = max(1, round(len(signer_order) * 0.2))
        val_count = max(1, round(len(signer_order) * 0.2))
        test_signers = set(signer_order[:test_count])
        val_signers = set(signer_order[test_count:test_count + val_count])
        train_signers = set(signer_order[test_count + val_count:])
        if not train_signers or not val_signers or not test_signers:
            raise ValueError("Not enough distinct signers for train/validation/test splits.")

        split_signers = {
            "train": train_signers,
            "val": val_signers,
            "test": test_signers,
        }
        vocabulary = sorted(label.upper() for label in expected_labels)
        label_to_index = {label: index for index, label in enumerate(vocabulary)}
        split_sequences = {split: [] for split in split_signers}
        split_labels = {split: [] for split in split_signers}
        processed = 0

        with zipfile.ZipFile(archive_path) as archive:
            for inner_name in sorted({example[2] for example in examples}):
                inner_examples = {
                    member: (label, signer)
                    for label, signer, example_archive, member in examples
                    if example_archive == inner_name
                }
                with tempfile.NamedTemporaryFile(suffix=".zip") as inner_file:
                    with archive.open(inner_name) as source:
                        while chunk := source.read(1024 * 1024):
                            inner_file.write(chunk)
                    inner_file.flush()

                    with zipfile.ZipFile(inner_file.name) as clips:
                        for member, (label, signer) in inner_examples.items():
                            split = next(
                                name for name, ids in split_signers.items() if signer in ids
                            )
                            with tempfile.NamedTemporaryFile(
                                suffix=os.path.splitext(member)[1]
                            ) as clip_file:
                                with clips.open(member) as source:
                                    while chunk := source.read(1024 * 1024):
                                        clip_file.write(chunk)
                                clip_file.flush()
                                sequence = self.process_video_clip(clip_file.name)

                            if not np.isfinite(sequence).all() or not np.any(sequence):
                                logger.warning("Skipping ISL clip with no usable landmarks: %s", member)
                                continue
                            split_sequences[split].append(sequence)
                            split_labels[split].append(label_to_index[label])
                            processed += 1
                            if processed % 25 == 0:
                                logger.info("Processed %d usable ISL emergency clips", processed)

        arrays = {}
        labels = {}
        for split in split_signers:
            if not split_sequences[split]:
                raise ValueError(f"No usable clips found in ISL {split} split.")
            arrays[f"X_{split}"] = np.asarray(split_sequences[split], dtype=np.float32)
            labels[f"y_{split}"] = np.asarray(split_labels[split], dtype=np.int32)
            logger.info("Usable ISL %s clips: %d", split, len(split_labels[split]))

        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            np.savez(
                cache_path,
                **arrays,
                **labels,
                vocabulary=np.asarray(vocabulary, dtype=np.str_),
                landmark_feature_version=np.asarray(
                    getattr(self, "landmark_feature_version", LANDMARK_FEATURE_VERSION)
                ),
                source_path=np.asarray(os.path.realpath(archive_path)),
                source_size=np.asarray(stat.st_size),
                source_mtime_ns=np.asarray(stat.st_mtime_ns),
                split_seed=np.asarray(seed),
                train_signers=np.asarray(sorted(train_signers), dtype=np.int32),
                val_signers=np.asarray(sorted(val_signers), dtype=np.int32),
                test_signers=np.asarray(sorted(test_signers), dtype=np.int32),
            )
            logger.info("Cached ISL emergency landmarks to %s", cache_path)

        return (
            arrays["X_train"],
            labels["y_train"],
            arrays["X_val"],
            labels["y_val"],
            arrays["X_test"],
            labels["y_test"],
            vocabulary,
        )

    def process_static_image(self, image_path):
        """
        Ingests a static RGB image, extracts wrist-centric landmark alignment,
        and expands temporally across T=60 frames for uniform sequence modeling.
        Returns:
            np.ndarray of shape (60, feature_dim)
        """
        frame = cv2.imread(image_path)
        if frame is None:
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)

        feat_63 = self.extractor.extract(frame)
        # Tile across sequence dimension T=60 with subtle natural stationary kinematic drift
        seq = np.tile(feat_63, (self.sequence_length, 1))
        # Add micro temporal kinematic jitter (variance 1e-4) to simulate real human hand stability
        noise = np.random.normal(0, 0.005, seq.shape).astype(np.float32)
        return seq + noise

    def augment_sequence(self, seq):
        """
        Data augmentation referencing sign-language-translator:
        Spatial 3D rotation, scale variation, and micro Gaussian joint jitter.
        """
        augmented = seq.copy()
        
        # 3D Euler angles (in degrees) for viewpoint invariance
        roll = np.random.uniform(-14.0, 14.0)
        pitch = np.random.uniform(-10.0, 10.0)
        yaw = np.random.uniform(-10.0, 10.0)
        
        r = np.radians(roll)
        p = np.radians(pitch)
        y = np.radians(yaw)
        
        Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], dtype=np.float32)
        Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]], dtype=np.float32)
        R = (Rz @ Ry @ Rx).T
        
        # Scale variation (distance from camera)
        scale = np.random.uniform(0.90, 1.10)
        
        for t in range(len(augmented)):
            p3d = augmented[t].reshape(-1, 3) @ R
            p3d = p3d * scale
            noise = np.random.normal(0, 0.005, p3d.shape).astype(np.float32)
            p3d += noise
            flat = p3d.flatten()
            norm = np.linalg.norm(flat) + 1e-6
            augmented[t] = flat / norm
            
        return augmented

    def generate_benchmark_corpus(self, samples_per_class=50, output_npy=None, output_h5=None):
        """
        Generates synthetic poses for smoke tests only. These samples are not
        real sign-language data and must not be used to report model accuracy.
        
        Returns:
            X: np.ndarray shape (N, 60, feature_dim)
            y: np.ndarray shape (N,)
        """
        X_list = []
        y_list = []

        logger.info(f"Generating benchmark corpus for {len(self.vocabulary)} classes ({samples_per_class} samples each)...")

        # Base finger configurations calibrated to ASL phonology
        # 5 fingers: Thumb, Index, Middle, Ring, Pinky
        for class_idx, label in enumerate(self.vocabulary):
            base_hand = self._synthesize_hand_pose(label)
            
            for s in range(samples_per_class):
                # Construct temporal trajectory over T=60 frames
                seq = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
                
                # Dynamic motion profiles for words vs static handshapes
                is_dynamic = label in ["HELLO", "THANK YOU", "YES", "NO", "PLEASE"]
                
                # Phase parameters
                omega = np.random.uniform(0.8, 1.2)
                phase_shift = np.random.uniform(0, np.pi)

                for t in range(self.sequence_length):
                    frame_pose = base_hand.copy()
                    
                    if is_dynamic:
                        # Dynamic trajectory (e.g. wave for HELLO, nod for YES, circle for PLEASE)
                        progress = t / float(self.sequence_length)
                        if label == "HELLO":
                            # Lateral wave oscillation of fingers and wrist
                            wave = np.sin(progress * 4 * np.pi * omega) * 0.15
                            frame_pose[5*3:9*3] += wave  # index moves
                            frame_pose[9*3:13*3] += wave # middle moves
                        elif label == "THANK YOU":
                            # Forward-downward trajectory from chin
                            disp = (1.0 - progress) * 0.18
                            frame_pose[2::3] += disp # z-axis extension
                        elif label == "YES":
                            # Vertical fist nodding motion
                            nod = np.sin(progress * 3 * np.pi * omega) * 0.12
                            frame_pose[1::3] += nod  # y-axis nod
                        elif label == "NO":
                            # Snap closure between thumb and index/middle
                            snap = np.sin(progress * 2 * np.pi) * 0.10
                            frame_pose[4*3:5*3] += snap
                        elif label == "PLEASE":
                            # Circular chest rubbing motion
                            cx = np.cos(progress * 2 * np.pi * omega) * 0.12
                            cy = np.sin(progress * 2 * np.pi * omega) * 0.12
                            frame_pose[0::3] += cx
                            frame_pose[1::3] += cy

                    # Add sample-specific natural variation
                    sample_noise = np.random.normal(0, 0.012, self.feature_dim).astype(np.float32)
                    frame_pose += sample_noise
                    
                    # Normalize wrist-centric norm
                    norm = np.linalg.norm(frame_pose) + 1e-6
                    seq[t] = frame_pose / norm

                X_list.append(seq)
                y_list.append(class_idx)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        # Shuffle
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]

        if output_npy:
            os.makedirs(os.path.dirname(os.path.abspath(output_npy)), exist_ok=True)
            np.save(output_npy, {"X": X, "y": y})
            logger.info(f"Saved corpus to {output_npy} (Shape: {X.shape})")

        if output_h5:
            os.makedirs(os.path.dirname(os.path.abspath(output_h5)), exist_ok=True)
            with h5py.File(output_h5, 'w') as h5f:
                h5f.create_dataset('X', data=X, compression='gzip')
                h5f.create_dataset('y', data=y, compression='gzip')
            logger.info(f"Saved HDF5 corpus to {output_h5}")

        return X, y

    def _synthesize_hand_pose(self, label):
        """
        Creates an approximate 21-joint pose for synthetic smoke-test examples.
        """
        pose = np.zeros((21, 3), dtype=np.float32)
        
        # 1. Joint 0: Wrist is origin (0, 0, 0)
        pose[0] = [0.0, 0.0, 0.0]

        # 2. Metacarpals: Rigid Palm Skeletal Arch (Joints 1, 2, 5, 9, 13, 17)
        # In MediaPipe, wrist is at (0,0), fingers extend upward in -y direction
        pose[1] = [-0.045, -0.075, -0.010]  # Thumb CMC
        pose[2] = [-0.090, -0.145, -0.020]  # Thumb MCP
        pose[5] = [-0.065, -0.240,  0.000]  # Index MCP
        pose[9] = [ 0.000, -0.260,  0.000]  # Middle MCP
        pose[13] = [ 0.055, -0.240,  0.000]  # Ring MCP
        pose[17] = [ 0.105, -0.200,  0.000]  # Pinky MCP

        # Finger bone lengths (L1: MCP->PIP, L2: PIP->DIP, L3: DIP->TIP)
        lengths = {
            'index':  (0.095, 0.065, 0.045),
            'middle': (0.105, 0.070, 0.048),
            'ring':   (0.095, 0.065, 0.045),
            'pinky':  (0.075, 0.050, 0.038),
        }
        
        # Base spread directions for extended fingers (normalized)
        spread_dirs = {
            'index':  np.array([-0.08, -0.99, 0.0], dtype=np.float32),
            'middle': np.array([ 0.00, -1.00, 0.0], dtype=np.float32),
            'ring':   np.array([ 0.08, -0.99, 0.0], dtype=np.float32),
            'pinky':  np.array([ 0.20, -0.98, 0.0], dtype=np.float32),
        }

        # Determine finger flexion states per sign:
        # 'ext' (extended), 'curl' (fist/curled), 'arc' (curved C/O), 'hook' (bent X), 'touch' (meeting thumb)
        f_states = {'index': 'ext', 'middle': 'ext', 'ring': 'ext', 'pinky': 'ext'}
        thumb_state = 'upright'
        
        # Precise ASL phonology calibration
        if label == "A":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'upright'  # Thumb straight up beside index MCP
        elif label == "B":
            f_states = {'index': 'ext', 'middle': 'ext', 'ring': 'ext', 'pinky': 'ext'}
            thumb_state = 'fold_palm'  # Thumb folded flat across palm
        elif label == "C":
            f_states = {'index': 'arc', 'middle': 'arc', 'ring': 'arc', 'pinky': 'arc'}
            thumb_state = 'arc'
        elif label == "D":
            f_states = {'index': 'ext', 'middle': 'touch', 'ring': 'touch', 'pinky': 'touch'}
            thumb_state = 'touch'
        elif label == "E":
            f_states = {'index': 'tight_curl', 'middle': 'tight_curl', 'ring': 'tight_curl', 'pinky': 'tight_curl'}
            thumb_state = 'tight_under'
        elif label == "F":
            f_states = {'index': 'touch', 'middle': 'ext', 'ring': 'ext', 'pinky': 'ext'}
            thumb_state = 'touch_index'
        elif label == "G":
            f_states = {'index': 'point_left', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'point_left'
        elif label == "H":
            f_states = {'index': 'point_left', 'middle': 'point_left', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_palm'
        elif label == "I":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'ext'}
            thumb_state = 'fold_over'
        elif label == "J":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'ext'}
            thumb_state = 'fold_over'
        elif label == "K":
            f_states = {'index': 'ext', 'middle': 'forward_ext', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'between_index_mid'
        elif label == "L":
            f_states = {'index': 'ext', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'horiz_90'
        elif label == "M":
            f_states = {'index': 'curl_over_thumb', 'middle': 'curl_over_thumb', 'ring': 'curl_over_thumb', 'pinky': 'curl'}
            thumb_state = 'under_three'
        elif label == "N":
            f_states = {'index': 'curl_over_thumb', 'middle': 'curl_over_thumb', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'under_two'
        elif label == "O":
            f_states = {'index': 'arc', 'middle': 'arc', 'ring': 'arc', 'pinky': 'arc'}
            thumb_state = 'arc_touch'
        elif label == "P":
            f_states = {'index': 'point_down', 'middle': 'forward_down', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'horiz_down'
        elif label == "Q":
            f_states = {'index': 'point_down', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'point_down'
        elif label == "R":
            f_states = {'index': 'cross_index', 'middle': 'cross_middle', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "S":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'  # Wrapped horizontally over the 4 curled fingers
        elif label == "T":
            f_states = {'index': 'curl_over_thumb', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'under_one'
        elif label == "U":
            f_states = {'index': 'parallel_up', 'middle': 'parallel_up', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "V":
            f_states = {'index': 'v_spread_left', 'middle': 'v_spread_right', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "W":
            f_states = {'index': 'w_left', 'middle': 'w_mid', 'ring': 'w_right', 'pinky': 'curl'}
            thumb_state = 'fold_over_pinky'
        elif label == "X":
            f_states = {'index': 'hook', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "Y":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'ext'}
            thumb_state = 'shaka_thumb'
        elif label == "Z":
            f_states = {'index': 'point_forward', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "YES":
            f_states = {'index': 'curl', 'middle': 'curl', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'fold_over'
        elif label == "HELLO":
            f_states = {'index': 'ext', 'middle': 'ext', 'ring': 'ext', 'pinky': 'ext'}
            thumb_state = 'open_spread'
        elif label == "THANK YOU":
            f_states = {'index': 'flat_chin', 'middle': 'flat_chin', 'ring': 'flat_chin', 'pinky': 'flat_chin'}
            thumb_state = 'fold_palm'
        elif label == "PLEASE":
            f_states = {'index': 'ext', 'middle': 'ext', 'ring': 'ext', 'pinky': 'ext'}
            thumb_state = 'open_spread'
        elif label == "NO":
            f_states = {'index': 'snap_beak', 'middle': 'snap_beak', 'ring': 'curl', 'pinky': 'curl'}
            thumb_state = 'snap_beak'

        # Articulate 4 fingers: Index (5-8), Middle (9-12), Ring (13-16), Pinky (17-20)
        finger_mcp_map = [('index', 5), ('middle', 9), ('ring', 13), ('pinky', 17)]
        for f_name, mcp_idx in finger_mcp_map:
            mcp = pose[mcp_idx]
            L1, L2, L3 = lengths[f_name]
            u = spread_dirs[f_name]
            st = f_states[f_name]

            if st == 'ext':
                pip = mcp + u * L1
                dip = pip + u * L2
                tip = dip + u * L3
            elif st in ('curl', 'curl_over_thumb'):
                pip = mcp + np.array([u[0] * 0.15, 0.045, -0.030], dtype=np.float32)
                dip = pip + np.array([0.0, 0.040, 0.020], dtype=np.float32)
                tip = dip + np.array([0.0, 0.025, 0.025], dtype=np.float32)
            elif st == 'tight_curl':
                pip = mcp + np.array([0.0, 0.035, -0.040], dtype=np.float32)
                dip = pip + np.array([0.0, 0.035, 0.015], dtype=np.float32)
                tip = dip + np.array([0.0, 0.020, 0.030], dtype=np.float32)
            elif st == 'arc':
                pip = mcp + np.array([u[0] * L1 * 0.7, -L1 * 0.7, -0.035], dtype=np.float32)
                dip = pip + np.array([u[0] * L2 * 0.5, -L2 * 0.5, -0.035], dtype=np.float32)
                tip = dip + np.array([u[0] * L3 * 0.2, L3 * 0.2, -0.020], dtype=np.float32)
            elif st == 'hook':
                pip = mcp + u * L1
                dip = pip + np.array([0.0, 0.045, -0.025], dtype=np.float32)
                tip = dip + np.array([0.0, 0.035, 0.015], dtype=np.float32)
            elif st == 'point_left':
                pip = mcp + np.array([-L1 * 0.9, -L1 * 0.2, 0.0], dtype=np.float32)
                dip = pip + np.array([-L2 * 0.9, -L2 * 0.2, 0.0], dtype=np.float32)
                tip = dip + np.array([-L3 * 0.9, -L3 * 0.2, 0.0], dtype=np.float32)
            elif st == 'point_down':
                pip = mcp + np.array([0.0, L1 * 0.9, 0.0], dtype=np.float32)
                dip = pip + np.array([0.0, L2 * 0.9, 0.0], dtype=np.float32)
                tip = dip + np.array([0.0, L3 * 0.9, 0.0], dtype=np.float32)
            elif st == 'point_forward':
                pip = mcp + np.array([0.0, -L1 * 0.5, -L1 * 0.8], dtype=np.float32)
                dip = pip + np.array([0.0, -L2 * 0.5, -L2 * 0.8], dtype=np.float32)
                tip = dip + np.array([0.0, -L3 * 0.5, -L3 * 0.8], dtype=np.float32)
            elif st == 'forward_ext':
                pip = mcp + np.array([u[0] * L1, -L1 * 0.7, -L1 * 0.5], dtype=np.float32)
                dip = pip + np.array([u[0] * L2, -L2 * 0.7, -L2 * 0.5], dtype=np.float32)
                tip = dip + np.array([u[0] * L3, -L3 * 0.7, -L3 * 0.5], dtype=np.float32)
            elif st == 'forward_down':
                pip = mcp + np.array([0.0, L1 * 0.6, -L1 * 0.6], dtype=np.float32)
                dip = pip + np.array([0.0, L2 * 0.6, -L2 * 0.6], dtype=np.float32)
                tip = dip + np.array([0.0, L3 * 0.6, -L3 * 0.6], dtype=np.float32)
            elif st == 'v_spread_left':
                uv = np.array([-0.18, -0.98, 0.0], dtype=np.float32)
                pip = mcp + uv * L1; dip = pip + uv * L2; tip = dip + uv * L3
            elif st == 'v_spread_right':
                uv = np.array([ 0.18, -0.98, 0.0], dtype=np.float32)
                pip = mcp + uv * L1; dip = pip + uv * L2; tip = dip + uv * L3
            elif st == 'parallel_up':
                up = np.array([-0.01 if f_name == 'index' else 0.01, -1.0, 0.0], dtype=np.float32)
                pip = mcp + up * L1; dip = pip + up * L2; tip = dip + up * L3
            elif st == 'w_left':
                uw = np.array([-0.16, -0.98, 0.0], dtype=np.float32)
                pip = mcp + uw * L1; dip = pip + uw * L2; tip = dip + uw * L3
            elif st == 'w_mid':
                uw = np.array([ 0.00, -1.00, 0.0], dtype=np.float32)
                pip = mcp + uw * L1; dip = pip + uw * L2; tip = dip + uw * L3
            elif st == 'w_right':
                uw = np.array([ 0.16, -0.98, 0.0], dtype=np.float32)
                pip = mcp + uw * L1; dip = pip + uw * L2; tip = dip + uw * L3
            elif st == 'cross_index':
                uc = np.array([ 0.06, -0.99, -0.01], dtype=np.float32)
                pip = mcp + uc * L1; dip = pip + uc * L2; tip = dip + uc * L3
            elif st == 'cross_middle':
                uc = np.array([-0.08, -0.99,  0.02], dtype=np.float32)
                pip = mcp + uc * L1; dip = pip + uc * L2; tip = dip + uc * L3
            elif st == 'touch':
                pip = mcp + np.array([u[0] * L1 * 0.6, -L1 * 0.6, -0.030], dtype=np.float32)
                dip = pip + np.array([u[0] * L2 * 0.4, -L2 * 0.4, -0.020], dtype=np.float32)
                tip = np.array([-0.05, -0.19, -0.03], dtype=np.float32)
            elif st == 'snap_beak':
                pip = mcp + np.array([0.0, -L1 * 0.7, -0.040], dtype=np.float32)
                dip = pip + np.array([0.0, -L2 * 0.4, -0.030], dtype=np.float32)
                tip = np.array([-0.04, -0.18, -0.04], dtype=np.float32)
            elif st == 'flat_chin':
                pip = mcp + np.array([u[0] * L1, -L1 * 0.8, -L1 * 0.3], dtype=np.float32)
                dip = pip + np.array([u[0] * L2, -L2 * 0.8, -L2 * 0.3], dtype=np.float32)
                tip = dip + np.array([u[0] * L3, -L3 * 0.8, -L3 * 0.3], dtype=np.float32)
            else:
                pip = mcp + u * L1; dip = pip + u * L2; tip = dip + u * L3

            pose[mcp_idx + 1] = pip
            pose[mcp_idx + 2] = dip
            pose[mcp_idx + 3] = tip

        # 3. Articulate Thumb: CMC (1), MCP (2), IP (3), TIP (4)
        mcp2 = pose[2]
        if thumb_state == 'upright':
            pose[3] = mcp2 + np.array([ 0.020, -0.055,  0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.015, -0.050,  0.010], dtype=np.float32)
        elif thumb_state == 'fold_palm':
            pose[3] = mcp2 + np.array([ 0.050, -0.010,  0.020], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.045,  0.005,  0.020], dtype=np.float32)
        elif thumb_state == 'fold_over':
            pose[3] = mcp2 + np.array([ 0.060, -0.035, -0.025], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.045, -0.015, -0.020], dtype=np.float32)
        elif thumb_state == 'fold_over_pinky':
            pose[3] = mcp2 + np.array([ 0.080, -0.030, -0.025], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.060, -0.010, -0.020], dtype=np.float32)
        elif thumb_state == 'horiz_90':
            pose[3] = mcp2 + np.array([-0.065,  0.005,  0.000], dtype=np.float32)
            pose[4] = pose[3] + np.array([-0.055,  0.005,  0.000], dtype=np.float32)
        elif thumb_state == 'shaka_thumb':
            pose[3] = mcp2 + np.array([-0.075, -0.010,  0.000], dtype=np.float32)
            pose[4] = pose[3] + np.array([-0.065, -0.010,  0.000], dtype=np.float32)
        elif thumb_state in ('arc', 'arc_touch'):
            pose[3] = mcp2 + np.array([-0.035, -0.045, -0.035], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.020, -0.035, -0.035], dtype=np.float32)
        elif thumb_state in ('touch', 'touch_index'):
            pose[3] = mcp2 + np.array([ 0.025, -0.040, -0.020], dtype=np.float32)
            pose[4] = np.array([-0.050, -0.190, -0.030], dtype=np.float32)
        elif thumb_state == 'between_index_mid':
            pose[3] = mcp2 + np.array([ 0.030, -0.050, -0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.020, -0.045, -0.010], dtype=np.float32)
        elif thumb_state == 'point_left':
            pose[3] = mcp2 + np.array([-0.055, -0.020, -0.015], dtype=np.float32)
            pose[4] = pose[3] + np.array([-0.045, -0.015, -0.015], dtype=np.float32)
        elif thumb_state == 'horiz_down':
            pose[3] = mcp2 + np.array([ 0.030,  0.040, -0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.020,  0.035, -0.010], dtype=np.float32)
        elif thumb_state == 'point_down':
            pose[3] = mcp2 + np.array([-0.035,  0.045, -0.015], dtype=np.float32)
            pose[4] = pose[3] + np.array([-0.025,  0.040, -0.015], dtype=np.float32)
        elif thumb_state == 'tight_under':
            pose[3] = mcp2 + np.array([ 0.040,  0.020,  0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.035,  0.020,  0.010], dtype=np.float32)
        elif thumb_state == 'under_one':
            pose[3] = mcp2 + np.array([ 0.030, -0.025,  0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.025, -0.015,  0.010], dtype=np.float32)
        elif thumb_state == 'under_two':
            pose[3] = mcp2 + np.array([ 0.055, -0.025,  0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.045, -0.015,  0.010], dtype=np.float32)
        elif thumb_state == 'under_three':
            pose[3] = mcp2 + np.array([ 0.080, -0.025,  0.010], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.065, -0.015,  0.010], dtype=np.float32)
        elif thumb_state == 'snap_beak':
            pose[3] = mcp2 + np.array([ 0.025, -0.040, -0.025], dtype=np.float32)
            pose[4] = np.array([-0.040, -0.180, -0.040], dtype=np.float32)
        elif thumb_state == 'open_spread':
            pose[3] = mcp2 + np.array([-0.055, -0.035,  0.000], dtype=np.float32)
            pose[4] = pose[3] + np.array([-0.045, -0.030,  0.000], dtype=np.float32)
        else:
            pose[3] = mcp2 + np.array([ 0.020, -0.050,  0.000], dtype=np.float32)
            pose[4] = pose[3] + np.array([ 0.020, -0.040,  0.000], dtype=np.float32)

        pts = pose.flatten()
        norm = np.linalg.norm(pts) + 1e-6
        return (pts / norm).astype(np.float32)
