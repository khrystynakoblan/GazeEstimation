import random
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MPIIGazeDataset(Dataset):
    def __init__(
        self,
        root_path: str,
        participants: list[str],
        split: str = "train",
        eyes: str = "both",
        samples_per_participant: int = 2000,
        transform: Optional[transforms.Compose] = None,
        gaze_ranges: Optional[dict] = None,
        img_mean: float = 0.0,
        img_std: float = 1.0,
    ):
        super().__init__()
        self.root_path = Path(root_path)
        self.participants = participants
        self.split = split
        self.eyes = eyes
        self.samples_per_participant = samples_per_participant
        self.transform = transform if split == "train" else None
        self.gaze_ranges = gaze_ranges
        self.img_mean = img_mean
        self.img_std = img_std

        self.images: list = []
        self.gazes_2d: list = []
        self.gazes_3d: list = []
        self.poses: list = []
        self.participant_ids: list = []
        self.eye_labels: list = []

        self._load_data()

        if self.gaze_ranges is None:
            self._compute_gaze_ranges()
        self._normalize_gazes()

    def _compute_gaze_ranges(self):
        gazes = np.array(self.gazes_2d, dtype=np.float32)
        yaw_abs   = max(abs(gazes[:, 0].min()), abs(gazes[:, 0].max())) * 1.05
        pitch_abs = max(abs(gazes[:, 1].min()), abs(gazes[:, 1].max())) * 1.05
        self.gaze_ranges = {
            "yaw":   (-yaw_abs,   yaw_abs),
            "pitch": (-pitch_abs, pitch_abs),
        }

    def _normalize_gazes(self):
        gazes = np.array(self.gazes_2d, dtype=np.float32)
        ymin, ymax = self.gaze_ranges["yaw"]
        pmin, pmax = self.gaze_ranges["pitch"]
        normalized = np.zeros_like(gazes)
        normalized[:, 0] = 2 * (gazes[:, 0] - ymin) / (ymax - ymin) - 1
        normalized[:, 1] = 2 * (gazes[:, 1] - pmin) / (pmax - pmin) - 1
        if self.split == "train":
            normalized = np.clip(normalized, -1.0, 1.0)
        self.gazes_2d_normalized = normalized

    def normalize_gaze(self, gaze_2d: np.ndarray) -> np.ndarray:
        ymin, ymax = self.gaze_ranges["yaw"]
        pmin, pmax = self.gaze_ranges["pitch"]
        normalized = np.array([
            2 * (gaze_2d[0] - ymin) / (ymax - ymin) - 1,
            2 * (gaze_2d[1] - pmin) / (pmax - pmin) - 1,
        ], dtype=np.float32)
        if self.split == "train":
            normalized = np.clip(normalized, -1.0, 1.0)
        return normalized

    def denormalize_gaze(self, norm_gaze: np.ndarray) -> np.ndarray:
        ymin, ymax = self.gaze_ranges["yaw"]
        pmin, pmax = self.gaze_ranges["pitch"]
        return np.array([
            (norm_gaze[0] + 1) * (ymax - ymin) / 2 + ymin,
            (norm_gaze[1] + 1) * (pmax - pmin) / 2 + pmin,
        ], dtype=np.float32)

    @staticmethod
    def gaze_3d_to_2d(gaze_3d: np.ndarray) -> np.ndarray:
        x, y, z = gaze_3d
        z = z if abs(z) >= 1e-6 else 1e-6 * np.sign(z)
        yaw   = np.arctan2(-x, -z)
        pitch = np.arctan2(-y, np.sqrt(x**2 + z**2))
        return np.array([yaw, pitch], dtype=np.float32)

    def _load_data(self):
        norm_path = self.root_path / "MPIIGaze" / "Data" / "Normalized"
        if not norm_path.exists():
            norm_path = self.root_path / "Data" / "Normalized"

        eyes_to_process = ["right", "left"] if self.eyes == "both" else [self.eyes]
        samples_per_eye = (
            self.samples_per_participant // 2
            if self.eyes == "both"
            else self.samples_per_participant
        )

        for participant_idx, participant in enumerate(self.participants):
            participant_path = norm_path / participant
            if not participant_path.exists():
                continue

            mat_files = sorted(participant_path.glob("*.mat"))
            participant_samples = 0
            eye_count = {"right": 0, "left": 0}

            for mat_file in mat_files:
                if participant_samples >= self.samples_per_participant:
                    break
                try:
                    data = sio.loadmat(str(mat_file))
                except Exception:
                    continue

                for eye_type in eyes_to_process:
                    if eye_count[eye_type] >= samples_per_eye:
                        continue
                    try:
                        eye_data = data["data"][0, 0][eye_type][0, 0]
                        if "image" not in eye_data.dtype.names:
                            continue

                        images   = eye_data["image"]
                        gazes_3d = eye_data["gaze"]
                        poses_3d = eye_data["pose"]
                        n = images.shape[0]
                        if n == 0:
                            continue

                        n_to_take = min(samples_per_eye - eye_count[eye_type], n)
                        chosen = np.random.choice(n, n_to_take, replace=False)

                        for i in chosen:
                            if participant_samples >= self.samples_per_participant:
                                break

                            img      = images[i].astype(np.float32)
                            gaze_3d  = gazes_3d[i].astype(np.float32)
                            pose_3d  = poses_3d[i].astype(np.float32)

                            if img.max() > 1.0:
                                img /= 255.0

                            if eye_type == "left":
                                img         = img[:, ::-1].copy()
                                gaze_3d[0]  = -gaze_3d[0]
                                pose_3d[1]  = -pose_3d[1]
                                pose_3d[2]  = -pose_3d[2]

                            gaze_2d = self.gaze_3d_to_2d(gaze_3d)
                            if np.any(np.isnan(gaze_2d)):
                                continue

                            self.images.append(img)
                            self.gazes_2d.append(gaze_2d)
                            self.gazes_3d.append(gaze_3d)
                            self.poses.append(pose_3d)
                            self.participant_ids.append(participant_idx)
                            self.eye_labels.append(0 if eye_type == "right" else 1)

                            participant_samples  += 1
                            eye_count[eye_type]  += 1

                    except Exception:
                        continue

        self.images          = np.array(self.images,          dtype=np.float32)
        self.gazes_2d        = np.array(self.gazes_2d,        dtype=np.float32)
        self.gazes_3d        = np.array(self.gazes_3d,        dtype=np.float32)
        self.poses           = np.array(self.poses,           dtype=np.float32)
        self.participant_ids = np.array(self.participant_ids,  dtype=np.int32)
        self.eye_labels      = np.array(self.eye_labels,       dtype=np.int32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img      = self.images[idx].copy()
        gaze_raw = self.gazes_2d[idx].copy()
        pose     = self.poses[idx].copy()

        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)

        img_tensor = (torch.from_numpy(img).float() - self.img_mean) / self.img_std

        if self.split == "train" and self.transform is not None:
            if random.random() > 0.5:
                img_tensor  = torch.flip(img_tensor, dims=[2])
                gaze_raw[0] = -gaze_raw[0]
                pose[1]     = -pose[1]
                pose[2]     = -pose[2]
            img_tensor = self.transform(img_tensor)

        gaze_norm = self.normalize_gaze(gaze_raw)

        metadata = {
            "participant_id": self.participant_ids[idx],
            "eye_label":      self.eye_labels[idx],
            "gaze_3d":        self.gazes_3d[idx].copy(),
            "gaze_original":  gaze_raw,
            "pose":           pose,
        }
        return (
            img_tensor,
            torch.from_numpy(gaze_norm).float(),
            torch.from_numpy(pose).float(),
            metadata,
        )
