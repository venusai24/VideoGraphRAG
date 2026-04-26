import numpy as np
import cv2
import torch
import logging
from typing import List, Optional
from PIL import Image
from ..models import NativeFrame

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Production feature extractor: CLIP embeddings, DINOv2 embeddings,
    Laplacian blur score, optical flow magnitude, and entity detection.
    Falls back to lightweight alternatives when heavyweight models are unavailable.
    """

    def __init__(self, clip_dim: int = 512, dino_dim: int = 384, device: str = None):
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim
        self.device = device or ("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
        self._prev_gray: Optional[np.ndarray] = None

        # ── CLIP ──────────────────────────────────────────────────────
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(self.device).eval()
            self._clip_proc = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_ready = True
            logger.info("CLIP model loaded on %s", self.device)
        except Exception as exc:
            logger.warning("CLIP unavailable (%s) – using random embeddings", exc)
            self._clip_ready = False

        # ── DINOv2 ────────────────────────────────────────────────────
        try:
            self._dino_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            ).to(self.device).eval()
            self._dino_ready = True
            logger.info("DINOv2 model loaded on %s", self.device)
        except Exception as exc:
            logger.warning("DINOv2 unavailable (%s) – using random embeddings", exc)
            self._dino_ready = False

        # ── Entity detection (YOLOv5 via torch.hub) ───────────────────
        try:
            from ultralytics import YOLO as UltralyticsYOLO
            self._yolo = UltralyticsYOLO("yolov8n.pt").to(self.device)
            self._yolo_ready = True
            logger.info("YOLOv8n entity detector loaded on %s", self.device)
        except Exception as exc:
            logger.warning("YOLO unavailable (%s) – entity list will be empty", exc)
            self._yolo_ready = False

    # ── public API ────────────────────────────────────────────────────

    def process_frame(self, frame_data: np.ndarray, timestamp: float) -> NativeFrame:
        """Extract all per-frame features from a BGR OpenCV frame."""
        clip_emb = self._extract_clip(frame_data)
        dino_emb = self._extract_dino(frame_data)
        blur_var = self._compute_blur(frame_data)
        flow_mag = self._compute_optical_flow(frame_data)
        entities = self._detect_entities(frame_data)

        return NativeFrame(
            timestamp=timestamp,
            clip_emb=clip_emb,
            dino_emb=dino_emb,
            blur_variance=blur_var,
            optical_flow_mag=flow_mag,
            entities=entities,
            frame_data=frame_data,
        )

    # ── CLIP ──────────────────────────────────────────────────────────

    def _extract_clip(self, bgr: np.ndarray) -> np.ndarray:
        if not self._clip_ready:
            emb = np.random.randn(self.clip_dim).astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        inputs = self._clip_proc(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            vision_out = self._clip_model.vision_model(pixel_values=pixel_values)
            # Project through the visual projection layer
            emb_tensor = self._clip_model.visual_projection(vision_out.pooler_output)
        emb = emb_tensor.squeeze().cpu().numpy().astype(np.float64)
        emb /= np.linalg.norm(emb) + 1e-8
        # project to configured dim
        if emb.shape[0] != self.clip_dim:
            emb = np.resize(emb, self.clip_dim)
            emb /= np.linalg.norm(emb) + 1e-8
        return emb

    # ── DINOv2 ────────────────────────────────────────────────────────

    def _extract_dino(self, bgr: np.ndarray) -> np.ndarray:
        if not self._dino_ready:
            emb = np.random.randn(self.dino_dim).astype(np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        # DINOv2 expects 224×224 normalised tensor
        from torchvision import transforms as T
        xform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        tensor = xform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self._dino_model(tensor)
        emb = emb.squeeze().cpu().numpy().astype(np.float64)
        emb /= np.linalg.norm(emb) + 1e-8
        if emb.shape[0] != self.dino_dim:
            emb = np.resize(emb, self.dino_dim)
            emb /= np.linalg.norm(emb) + 1e-8
        return emb

    # ── Blur (Laplacian variance) ─────────────────────────────────────

    @staticmethod
    def _compute_blur(bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Optical flow (Farneback dense) ────────────────────────────────

    def _compute_optical_flow(self, bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # down-sample for speed
        small = cv2.resize(gray, (160, 120))
        if self._prev_gray is None:
            self._prev_gray = small
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, small,
            None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self._prev_gray = small
        return float(np.mean(mag))

    # ── Entity detection ──────────────────────────────────────────────

    def _detect_entities(self, bgr: np.ndarray) -> List[dict]:
        if not self._yolo_ready:
            return []
        results = self._yolo(bgr, verbose=False, conf=0.35)
        detections = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                detections.append({
                    "bbox": [float(v) for v in xyxy],
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": r.names.get(cls_id, str(cls_id)),
                })
        return detections
