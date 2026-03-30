"""
Centralized configuration for the Video RAG preprocessing pipeline.

All tuneable parameters are defined here so that every module
draws from a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """Configuration for the video preprocessing pipeline."""

    # ── Frame Sampling ──────────────────────────────────────────────
    fps_sample: float = 12.0  # Target sampling rate (frames per second)

    # ── Blur Filtering ──────────────────────────────────────────────
    blur_threshold: float = 120.0  # Laplacian variance below this → blurry

    # ── Scene Detection ─────────────────────────────────────────────
    scene_sim_threshold: float = 0.80  # Cosine‑similarity drop → new scene

    # ── Similarity Grouping ─────────────────────────────────────────
    sim_t1: float = 0.92  # Local continuity threshold
    sim_t2: float = 0.88  # Group anchor constraint threshold

    # ── Frame Selection ─────────────────────────────────────────────
    w_blur: float = 0.6       # Weight for blur score in selection
    w_centrality: float = 0.4  # Weight for centrality in selection

    # ── Embedding ───────────────────────────────────────────────────
    batch_size: int = 32       # Batch size for model inference
    device: str = "cuda"       # PyTorch device ("cuda" or "cpu")

    # ── CLIP Model ──────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-base-patch32"

    # ── DINOv2 Model ────────────────────────────────────────────────
    dino_model_name: str = "facebook/dinov2-base"

    # ── Storage ─────────────────────────────────────────────────────
    output_dir: Path = field(default_factory=lambda: Path("output"))
    save_images: bool = True
    image_format: str = "jpg"
    image_quality: int = 95  # JPEG quality (1‑100)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
