"""
Embedding extractor – produces CLIP and DINOv2 embeddings in batches.

Models are loaded lazily on first call and kept in memory for reuse.
All embeddings are L2‑normalised before being returned.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)

from video_rag_preprocessing.config import PipelineConfig
from video_rag_preprocessing.utils.image_utils import to_rgb

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extracts CLIP and DINOv2 embeddings from BGR frames.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration (model names, batch size, device).
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.device)
        logger.info("Embedding device: %s", self.device)

        # ── CLIP ────────────────────────────────────────────────────
        logger.info("Loading CLIP model: %s", config.clip_model_name)
        self.clip_model = CLIPModel.from_pretrained(config.clip_model_name).to(
            self.device
        )
        self.clip_processor = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.clip_model.eval()

        # ── DINOv2 ──────────────────────────────────────────────────
        logger.info("Loading DINOv2 model: %s", config.dino_model_name)
        self.dino_model = AutoModel.from_pretrained(config.dino_model_name).to(
            self.device
        )
        self.dino_processor = AutoImageProcessor.from_pretrained(config.dino_model_name)
        self.dino_model.eval()

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def extract(
        self, frames: List[np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """Extract CLIP + DINOv2 embeddings for a list of BGR frames.

        Parameters
        ----------
        frames : list[np.ndarray]
            BGR images (any resolution).

        Returns
        -------
        list[dict[str, np.ndarray]]
            Each dict has keys ``"embedding_clip"`` and ``"embedding_dino"``,
            both 1‑D float32 NumPy arrays (L2‑normalised).
        """
        pil_images = [Image.fromarray(to_rgb(f)) for f in frames]

        clip_embs = self._extract_clip(pil_images)
        dino_embs = self._extract_dino(pil_images)

        results: List[Dict[str, np.ndarray]] = []
        for c, d in zip(clip_embs, dino_embs):
            results.append({
                "embedding_clip": c,
                "embedding_dino": d,
            })
        return results

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        """L2‑normalise each row of *vectors* in‑place."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return vectors / norms

    def _extract_clip(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Batch CLIP image embedding extraction."""
        all_embs: List[np.ndarray] = []
        bs = self.config.batch_size

        for start in tqdm(
            range(0, len(images), bs),
            desc="CLIP embeddings",
            unit="batch",
        ):
            batch = images[start : start + bs]
            inputs = self.clip_processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)

            # Handle different return types based on transformers version
            if torch.is_tensor(outputs):
                features = outputs
            elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
                features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs[0]  # Fallback to tuple indexing

            embs = features.cpu().numpy().astype(np.float32)
            all_embs.append(embs)

        concatenated = np.concatenate(all_embs, axis=0)
        return list(self._normalize(concatenated))

    def _extract_dino(self, images: List[Image.Image]) -> List[np.ndarray]:
        """Batch DINOv2 CLS‑token embedding extraction."""
        all_embs: List[np.ndarray] = []
        bs = self.config.batch_size

        for start in tqdm(
            range(0, len(images), bs),
            desc="DINOv2 embeddings",
            unit="batch",
        ):
            batch = images[start : start + bs]
            inputs = self.dino_processor(
                images=batch, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.dino_model(**inputs)

            # CLS token is the first token of last_hidden_state
            cls_embs = outputs.last_hidden_state[:, 0, :]
            embs = cls_embs.cpu().numpy().astype(np.float32)
            all_embs.append(embs)

        concatenated = np.concatenate(all_embs, axis=0)
        return list(self._normalize(concatenated))
