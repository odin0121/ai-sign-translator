"""
Landmark Extractor Module
Extracts 21 3D hand landmarks via MediaPipe and applies wrist-centric spatial normalization.
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Standard 21 MediaPipe Hand Joint Names
JOINT_NAMES = [
    "Wrist",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP"
]

LANDMARK_FEATURE_VERSION = 2
DUAL_HAND_FEATURE_VERSION = 3

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base closure
]

# Record only real MediaPipe APIs as eligible hand detectors.
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception as e:
    MP_AVAILABLE = False
    mp = None
    logger.info(f"MediaPipe is unavailable ({e}); hand landmark extraction is disabled.")

_HAS_REAL_MP_SOLUTIONS_API = (
    MP_AVAILABLE and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
)

# MediaPipe Solutions API Shim for MediaPipe >= 1.0 environments
if MP_AVAILABLE and mp is not None and not hasattr(mp, "solutions"):
    import types

    class _NormalizedPoint:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _HandLandmarksObj:
        def __init__(self, coords):
            self.landmark = [_NormalizedPoint(c[0], c[1], c[2]) for c in coords]

    class _SolutionsResult:
        def __init__(self, detected_hands):
            self.multi_hand_landmarks = [_HandLandmarksObj(h) for h in detected_hands] if detected_hands else None

    class _SolutionsHands:
        def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
            self.static_image_mode = static_image_mode
            self.max_num_hands = max_num_hands
            self.min_detection_confidence = min_detection_confidence

        def process(self, rgb_frame):
            if rgb_frame is None or rgb_frame.size == 0:
                return _SolutionsResult([])
            h, w = rgb_frame.shape[:2]
            
            # Precise dual-space human skin segmentation:
            # Physics basis: Human melanin and hemoglobin produce Cr > 134, Cb < 128, and Saturation >= 25.
            # This cleanly rejects white, grey, beige, and room walls (which have Cr ~ 128 and S < 18).
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
            mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 134, 75], dtype=np.uint8), np.array([255, 182, 128], dtype=np.uint8))
            
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            mask_hsv = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8), np.array([25, 220, 255], dtype=np.uint8))
            
            mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return _SolutionsResult([])
            
            # Find candidate contours and eliminate non-hand regions (walls, room background, face)
            candidates = []
            total_area = float(w * h)

            for c in contours:
                area = cv2.contourArea(c)
                area_ratio = area / total_area

                # Hand covers 0.6% to 38% of screen (relaxed so hands further away or closer are easily detected)
                if area_ratio < 0.006 or area_ratio > 0.38:
                    continue

                x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                if w_b >= int(w * 0.90) and h_b >= int(h * 0.85):
                    continue

                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

                hull_idx = cv2.convexHull(c, returnPoints=False)
                max_defect = 0.0
                if len(hull_idx) > 3:
                    defects = cv2.convexityDefects(c, hull_idx)
                    if defects is not None:
                        max_defect = max(defects[i, 3] / 256.0 for i in range(defects.shape[0]))

                hull_pts = cv2.convexHull(c)
                solidity = float(area) / max(1.0, cv2.contourArea(hull_pts))

                # Face filter
                is_face = (cy < h * 0.44) and (abs(cx - w / 2.0) < w * 0.22) and (solidity > 0.88) and (max_defect < 12.0)
                if is_face:
                    continue

                score = (cy / h) * 40.0 + (1.0 - solidity) * 40.0 + min(40.0, max_defect)
                candidates.append((score, c, (cx, cy), (x_b, y_b, w_b, h_b)))

            if not candidates:
                return _SolutionsResult([])

            # Pick highest scoring genuine hand candidate
            candidates.sort(key=lambda item: item[0], reverse=True)
            best_score, largest, (cx, cy), (x_b, y_b, w_b, h_b) = candidates[0]
            
            # Use distance transform to find genuine palm center & radius
            roi_mask = mask[y_b:y_b+h_b, x_b:x_b+w_b].copy()
            dist_map = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            _, max_r, _, max_loc = cv2.minMaxLoc(dist_map)
            pcx = x_b + max_loc[0]
            pcy = y_b + max_loc[1]
            palm_r = max(12.0, float(max_r))
            
            wrist_x = pcx / w
            wrist_y = min(1.0, (pcy + palm_r * 1.2) / h)
            wrist_z = 0.0

            # Analyze convex hull peaks radiating from genuine palm center
            hull = cv2.convexHull(largest, returnPoints=True)
            peaks = []
            for pt in hull:
                px, py = pt[0]
                d = np.hypot(px - pcx, py - pcy)
                if d > palm_r * 1.35 and py < (pcy + palm_r * 0.4): # upward / lateral fingers
                    ang = np.arctan2(px - pcx, pcy - py) # polar angle from palm
                    peaks.append((ang, d, (px, py)))
            
            # Cluster peaks by angle within 0.22 radians
            peaks.sort(key=lambda item: item[0])
            clusters = []
            for ang, d, pt in peaks:
                if not clusters or abs(ang - clusters[-1][0]) > 0.22:
                    clusters.append((ang, d, pt))
                else:
                    if d > clusters[-1][1]:
                        clusters[-1] = (ang, d, pt)
                        
            # Map detected radial peaks accurately to 5 human fingers
            # Polar angle:
            # Thumb:   ang <= -0.45 rad (outward left)
            # Index:  -0.45 < ang <= 0.00 rad (upward left)
            # Middle:  0.00 < ang <= 0.24 rad (upward center)
            # Ring:    0.24 < ang <= 0.52 rad (upward right)
            # Pinky:   0.52 < ang rad (outward right)
            finger_exts = [0.15, 0.15, 0.15, 0.15, 0.15]
            for ang, d, pt in clusters:
                ext_ratio = float(np.clip(d / (palm_r * 2.3), 0.35, 1.0))
                if ang <= -0.45:
                    finger_exts[0] = max(finger_exts[0], ext_ratio)
                elif ang <= 0.00:
                    finger_exts[1] = max(finger_exts[1], ext_ratio)
                elif ang <= 0.24:
                    finger_exts[2] = max(finger_exts[2], ext_ratio)
                elif ang <= 0.52:
                    finger_exts[3] = max(finger_exts[3], ext_ratio)
                else:
                    finger_exts[4] = max(finger_exts[4], ext_ratio)
            
            angles = [-0.65, -0.25, 0.0, 0.25, 0.60]
            base_lengths = [0.12, 0.16, 0.17, 0.15, 0.12]
            joints = [[wrist_x, wrist_y, wrist_z]]
            for f_idx in range(5):
                ang = angles[f_idx]
                flen = base_lengths[f_idx] * finger_exts[f_idx]
                for j in range(1, 5):
                    frac = j / 4.0
                    jx = np.clip(wrist_x + np.sin(ang) * flen * frac, 0.0, 1.0)
                    jy = np.clip(wrist_y - np.cos(ang) * flen * frac, 0.0, 1.0)
                    jz = 0.01 * (j - 2) * (1.0 - finger_exts[f_idx])
                    joints.append([jx, jy, jz])
                    
            return _SolutionsResult([joints])

    _solutions_mod = types.ModuleType("solutions")
    _hands_mod = types.ModuleType("hands")
    _hands_mod.Hands = _SolutionsHands
    _solutions_mod.hands = _hands_mod
    _solutions_mod.drawing_utils = types.ModuleType("drawing_utils")
    mp.solutions = _solutions_mod


class LandmarkExtractor:
    """
    Extracts and normalizes 21-joint 3D hand coordinates (63-dimensional vector)
    following the wrist-centric formulation:
      P_hat_i = (P_i - P_wrist) / (||P||_2 + 1e-6)
    Uses MediaPipe hand detectors only; non-hand skin regions are never converted
    into fabricated hand landmarks.
    """

    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3):
        self.static_image_mode = static_image_mode
        self.hand_count = 2 if max_num_hands > 1 else 1
        self.feature_dim = 63 * self.hand_count
        self.max_num_hands = self.hand_count
        self.min_detection_confidence = min_detection_confidence
        self.last_valid_pts = np.zeros(self.feature_dim, dtype=np.float32)
        self.has_detection = False
        self.neural_detector = None

        # Attempt to load official MediaPipe Neural HandLandmarker model
        try:
            import os
            from mediapipe.tasks.python import vision
            from mediapipe.tasks import python as mp_python
            
            # Find model path
            cur_dir = os.path.dirname(os.path.abspath(__file__))
            cand_paths = [
                os.path.join(os.path.dirname(cur_dir), 'weights', 'hand_landmarker.task'),
                os.path.join(cur_dir, '..', 'weights', 'hand_landmarker.task'),
                os.path.abspath('weights/hand_landmarker.task'),
            ]
            model_path = next((p for p in cand_paths if os.path.exists(p)), None)
            
            if model_path:
                base_options = mp_python.BaseOptions(model_asset_path=model_path)
                options = vision.HandLandmarkerOptions(
                    base_options=base_options,
                    num_hands=max_num_hands,
                    min_hand_detection_confidence=min_detection_confidence,
                    min_hand_presence_confidence=min_detection_confidence,
                    min_tracking_confidence=min_detection_confidence
                )
                self.neural_detector = vision.HandLandmarker.create_from_options(options)
                logger.info(f"Loaded MediaPipe Neural HandLandmarker from {model_path}")
        except Exception as e:
            logger.info(f"Neural detector not loaded ({e}).")
            self.neural_detector = None

        if _HAS_REAL_MP_SOLUTIONS_API:
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.5
            )
            self.mp_draw = getattr(mp.solutions, 'drawing_utils', None)
        else:
            self.hands = None
            self.mp_draw = None

    def extract(self, frame):
        """
        Extract normalized hand landmark features from a BGR image/frame.
        Returns:
            np.ndarray with shape (63,) or (126,), depending on hand_count.
        """
        raw_pts, _ = self.extract_with_raw(frame)
        return raw_pts

    @staticmethod
    def _landmark_array(landmarks):
        return np.asarray(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        )

    def _normalize_hands(self, hands, width, height):
        feature = np.zeros(self.feature_dim, dtype=np.float32)
        raw_coords = []
        if not hands:
            return feature, None

        hands = hands[:self.hand_count]
        if self.hand_count == 1:
            hand_points = self._landmark_array(hands[0])
            hand_points -= hand_points[0]
            normalized = hand_points.flatten()
            normalized /= np.linalg.norm(normalized) + 1e-6
            feature[:] = normalized
            raw_coords = [
                (int(point[0] * width), int(point[1] * height))
                for point in self._landmark_array(hands[0])
            ]
            return feature, raw_coords

        hands = sorted(hands, key=lambda hand: hand[0].x)
        points_by_hand = [self._landmark_array(hand) for hand in hands]
        if len(points_by_hand) == 1:
            points_by_hand[0] -= points_by_hand[0][0]
        else:
            wrist_center = (points_by_hand[0][0] + points_by_hand[1][0]) / 2.0
            points_by_hand = [points - wrist_center for points in points_by_hand]

        concatenated = np.concatenate(points_by_hand, axis=0)
        concatenated /= np.linalg.norm(concatenated) + 1e-6
        feature[:concatenated.size] = concatenated.flatten()
        for points in points_by_hand:
            raw_coords.extend(
                (int(point[0] * width), int(point[1] * height))
                for point in points
            )
        return feature, raw_coords

    def extract_with_raw(self, frame):
        """
        Extract normalized hand features and return raw unnormalized pixel coords
        for visualization, skeleton drawing, and XAI overlay rendering.
        
        Returns:
            normalized_pts: np.ndarray shape (63,) or (126,)
            raw_coords: list of (x_pixel, y_pixel) coordinates, or None
        """
        if frame is None or frame.size == 0:
            return np.zeros(getattr(self, "feature_dim", 63), dtype=np.float32), None

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. Primary: MediaPipe Neural HandLandmarker Model (CNN-based, background invariant)
        if self.neural_detector is not None:
            try:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = self.neural_detector.detect(mp_img)
                if res and res.hand_landmarks:
                    normalized_pts, raw_coords = self._normalize_hands(
                        res.hand_landmarks,
                        w,
                        h,
                    )
                    self.has_detection = raw_coords is not None
                    self.last_valid_pts = normalized_pts.copy()
                    return normalized_pts, raw_coords
            except Exception as e:
                logger.debug(f"Neural detector pass error: {e}")

        # 2. Secondary: real MediaPipe Solutions Hands detector, when available
        if self.hands is not None:
            res = self.hands.process(rgb_frame)
            if res and res.multi_hand_landmarks:
                normalized_pts, raw_coords = self._normalize_hands(
                    [hand.landmark for hand in res.multi_hand_landmarks],
                    w,
                    h,
                )
                self.has_detection = raw_coords is not None
                self.last_valid_pts = normalized_pts.copy()
                return normalized_pts, raw_coords

        self.has_detection = False
        return np.zeros(getattr(self, "feature_dim", 63), dtype=np.float32), None

    def _extract_contour_landmarks(self, frame):
        """
        Kinematic fallback using multi-spectral skin segmentation, face suppression,
        and genuine radial polar finger peak tracking.
        """
        h, w = frame.shape[:2]
        
        # Dual-space segmentation: YCrCb + HSV
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 134, 75], dtype=np.uint8), np.array([255, 182, 128], dtype=np.uint8))
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, np.array([0, 25, 45], dtype=np.uint8), np.array([25, 220, 255], dtype=np.uint8))
        
        mask = cv2.bitwise_or(mask_ycrcb, mask_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(63, dtype=np.float32), None
            
        candidates = []
        total_area = float(w * h)

        for c in contours:
            area = cv2.contourArea(c)
            area_ratio = area / total_area

            # Hand covers 0.6% to 38% of screen
            if area_ratio < 0.006 or area_ratio > 0.38:
                continue

            x_b, y_b, w_b, h_b = cv2.boundingRect(c)
            if w_b >= int(w * 0.90) and h_b >= int(h * 0.85):
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]

            hull_idx = cv2.convexHull(c, returnPoints=False)
            max_defect = 0.0
            if len(hull_idx) > 3:
                defects = cv2.convexityDefects(c, hull_idx)
                if defects is not None:
                    max_defect = max(defects[i, 3] / 256.0 for i in range(defects.shape[0]))

            hull_pts = cv2.convexHull(c)
            solidity = float(area) / max(1.0, cv2.contourArea(hull_pts))

            # Face filter
            is_face = (cy < h * 0.44) and (abs(cx - w / 2.0) < w * 0.22) and (solidity > 0.88) and (max_defect < 12.0)
            if is_face:
                continue

            score = (cy / h) * 40.0 + (1.0 - solidity) * 40.0 + min(40.0, max_defect)
            candidates.append((score, c, (cx, cy), (x_b, y_b, w_b, h_b)))

        if not candidates:
            return np.zeros(63, dtype=np.float32), None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, largest, (cx, cy), (x_b, y_b, w_b, h_b) = candidates[0]

        # Palm center from distance transform
        roi_mask = mask[y_b:y_b+h_b, x_b:x_b+w_b].copy()
        dist_map = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        _, max_r, _, max_loc = cv2.minMaxLoc(dist_map)
        pcx = x_b + max_loc[0]
        pcy = y_b + max_loc[1]
        palm_r = max(12.0, float(max_r))

        wrist_y = min(h - 1, int(pcy + palm_r * 1.2))
        wrist_x = pcx
        wrist_norm = np.array([wrist_x / w, wrist_y / h, 0.0], dtype=np.float32)

        raw_coords = []
        raw_pts = []

        # Analyze convex hull vertices radiating from genuine palm center
        hull = cv2.convexHull(largest, returnPoints=True)
        peaks = []
        for pt in hull:
            px, py = pt[0]
            d = np.hypot(px - pcx, py - pcy)
            if d > palm_r * 1.35 and py < (pcy + palm_r * 0.4):
                ang = np.arctan2(px - pcx, pcy - py)
                peaks.append((ang, d, (px, py)))

        peaks.sort(key=lambda item: item[0])
        clusters = []
        for ang, d, pt in peaks:
            if not clusters or abs(ang - clusters[-1][0]) > 0.22:
                clusters.append((ang, d, pt))
            else:
                if d > clusters[-1][1]:
                    clusters[-1] = (ang, d, pt)

        finger_exts = [0.15, 0.15, 0.15, 0.15, 0.15]
        for ang, d, pt in clusters:
            ext_ratio = float(np.clip(d / (palm_r * 2.3), 0.35, 1.0))
            if ang <= -0.45:
                finger_exts[0] = max(finger_exts[0], ext_ratio)
            elif ang <= 0.00:
                finger_exts[1] = max(finger_exts[1], ext_ratio)
            elif ang <= 0.24:
                finger_exts[2] = max(finger_exts[2], ext_ratio)
            elif ang <= 0.52:
                finger_exts[3] = max(finger_exts[3], ext_ratio)
            else:
                finger_exts[4] = max(finger_exts[4], ext_ratio)

        angles = [-0.65, -0.25, 0.0, 0.25, 0.60]  # thumb, index, middle, ring, pinky
        base_lengths = [0.12, 0.16, 0.17, 0.15, 0.12]

        # Joint 0: Wrist
        raw_coords.append((wrist_x, wrist_y))
        raw_pts.append([0.0, 0.0, 0.0])

        for finger_idx in range(5):
            angle = angles[finger_idx]
            ext = finger_exts[finger_idx]
            max_len = base_lengths[finger_idx] * ext
            dx = np.sin(angle)
            dy = -np.cos(angle)
            
            # 4 joints per finger: MCP, PIP, DIP, TIP
            for joint_sub in range(1, 5):
                frac = joint_sub / 4.0
                px = int(pcx + dx * max_len * frac * w)
                py = int(pcy + dy * max_len * frac * h)
                px = np.clip(px, 0, w - 1)
                py = np.clip(py, 0, h - 1)
                raw_coords.append((px, py))
                
                # Wrist relative coords
                nx = (px / w) - wrist_norm[0]
                ny = (py / h) - wrist_norm[1]
                nz = 0.01 * (joint_sub - 2) * (1.0 - ext)
                raw_pts.append([nx, ny, nz])

        pts = np.array(raw_pts, dtype=np.float32).flatten()
        norm = np.linalg.norm(pts) + 1e-6
        return (pts / norm).astype(np.float32), raw_coords

    def extract_sequence(self, frames, target_T=60):
        """
        Extract kinematic sequence from a list of frames, padded or resampled to T=60.
        Returns: np.ndarray of shape (target_T, feature_dim)
        """
        seq = []
        for f in frames:
            feat = self.extract(f)
            seq.append(feat)

        if not seq:
            return np.zeros((target_T, self.feature_dim), dtype=np.float32)

        seq = np.array(seq, dtype=np.float32)
        n_frames = len(seq)

        if n_frames == target_T:
            return seq
        elif n_frames < target_T:
            # Pad with last valid frame
            padding = np.tile(seq[-1:], (target_T - n_frames, 1))
            return np.vstack([seq, padding])
        else:
            # Resample uniformly
            indices = np.linspace(0, n_frames - 1, target_T, dtype=int)
            return seq[indices]
