import csv
from pathlib import Path
from typing import Sequence, Dict

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.manifest import load_manifest_csv, SampleManifest
from preprocessing.transforms import DeterministicPreprocessor
from preprocessing.labels import encode_labels, TARGET_LABELS

class CheXpertDataset(Dataset):
    """PyTorch Dataset for CheXpert using pre-generated manifest CSV.

    Parameters
    ----------
    manifest_path: Path
        Path to a manifest CSV file (train, val, or test).
    preprocess: DeterministicPreprocessor
        Preprocessor instance to apply deterministic transforms.
    policy: str, default "u_ignore"
        Uncertainty handling policy passed to ``encode_labels``.
    training: bool, default False
        If True, the dataset will also apply the optional ``TrainingAugmentor``
        (currently not used in Phase 2). This flag exists for future extensibility.
    """

    def __init__(self, manifest_path: Path, preprocess: DeterministicPreprocessor, policy: str = "u_ignore", training: bool = False) -> None:
        self.samples: Sequence[SampleManifest] = load_manifest_csv(manifest_path)
        self.preprocess = preprocess
        self.policy = policy
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        # Load and preprocess image (numpy -> torch)
        img_np: np.ndarray = self.preprocess(sample.image_path)
        img_tensor = torch.from_numpy(img_np).float()
        # Encode raw labels according to policy, producing targets and masks dicts
        raw_labels = sample.raw_labels
        targets_dict, masks_dict = encode_labels(raw_labels, policy=self.policy)
        # Preserve order defined by TARGET_LABELS
        target_vec = torch.tensor([targets_dict[d] for d in TARGET_LABELS], dtype=torch.float32)
        mask_vec = torch.tensor([masks_dict[d] for d in TARGET_LABELS], dtype=torch.float32)
        return {"image": img_tensor, "target": target_vec, "mask": mask_vec}
