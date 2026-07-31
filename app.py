"""
app.py v6 — DSS Analisis Kemiripan Logo Merek Dagang
Universitas Widyatama | Ghania Fazila (41122100060)

Perbaikan utama:
- Path aplikasi dapat mengikuti environment notebook.
- Checkpoint dimuat dengan weights_only=True dan divalidasi.
- Database embedding/HSV/metadata divalidasi penuh.
- Retrieval menampilkan skor embedding, HSV, dan skor akhir.
- Bobot fusion alpha dapat diatur; tidak ada penghapusan kandidat berdasarkan skor.
- Filter Kelas Nice mendukung satu atau beberapa kelas.
- Resolusi path gambar mengutamakan relative path dan menghindari filename ambigu.
- Upload gambar divalidasi dan transparansi dikompositkan secara konsisten.
- Top-K efektif menyesuaikan jumlah data hasil filter.
- Istilah keputusan dibuat lebih hati-hati karena sistem merupakan DSS.
- Explainability utama menggunakan Occlusion Sensitivity terhadap skor fusion.
- EigenGradCAM tetap tersedia sebagai penjelasan tambahan khusus cabang CNN.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageOps, UnidentifiedImageError
from pytorch_grad_cam import EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from skimage.feature import hog
from torchvision import models


# =============================================================================
# KONFIGURASI
# =============================================================================

APP_VERSION = "6.0"

# =============================================================================
# PALETTE UI DJKI
# =============================================================================

COLOR_NAVY      = "#1D2D5C"
COLOR_BLUE      = "#30498A"
COLOR_YELLOW    = "#FFCC2B"
COLOR_ORANGE    = "#EF7D14"
COLOR_RED_WARN  = "#C0392B"

COLOR_BACKGROUND    = "#F5F7FA"
COLOR_WHITE     = "#FFFFFF"
COLOR_TEXT      = "#1F2937"
COLOR_MUTED     = "#6B7280"

DEFAULT_BASE_DIR    = Path(__file__).resolve().parent
BASE_DIR    = Path(os.getenv("DSS_APP_DIR", str(DEFAULT_BASE_DIR))).resolve()
MODEL_DIR   = Path(os.getenv("DSS_MODEL_DIR", str(BASE_DIR))).resolve()
IMAGES_DIR  = Path(
    os.getenv("DSS_PDKI_DIR", str(BASE_DIR / "pdki_images"))
).resolve()

MODEL_PATH  = MODEL_DIR / "hybrid_best.pt"
EMBEDS_PATH = MODEL_DIR / "pdki_embeddings.npy"
HSV_PATH    = MODEL_DIR / "pdki_hsv.npy"
METADATA_PATH = MODEL_DIR / "pdki_metadata.json"

IMG_SIZE    = 224
EMBED_DIM   = 256

HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

HSV_BINS    = [16, 8, 8]
HSV_DIM     = sum(HSV_BINS)

DEFAULT_TOP_K = 10

# Nilai awal operasional. Nilai final untuk skripsi sebaiknya ditentukan
# melalui evaluasi validation set.
DEFAULT_ALPHA_FUSION = float(os.getenv("DSS_ALPHA_FUSION", "1.0"))
DEFAULT_SIMILARITY_THRESHOLD = float(
    os.getenv("DSS_SIMILARITY_THRESHOLD", "0.6")
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Konfigurasi default Occlusion Sensitivity. Nilai ini dipilih agar masih
# realistis dijalankan pada CPU Google Colab.
DEFAULT_OCCLUSION_PATCH_SIZE = 56
DEFAULT_OCCLUSION_STRIDE = 28
DEFAULT_OCCLUSION_BATCH_SIZE = 8


# =============================================================================
# PREPROCESSING
# =============================================================================

class LetterboxResize:
    """Resize dengan mempertahankan aspek rasio dan padding hitam.

    Padding hitam dipertahankan agar konsisten dengan pipeline model/database
    yang digunakan sebelumnya.
    """

    def __init__(self, size: int = IMG_SIZE):
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert("RGB")
        width, height = img.size

        if width <= 0 or height <= 0:
            raise ValueError("Dimensi gambar tidak valid.")

        scale = self.size / max(width, height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        resized = img.resize(
            (resized_width, resized_height),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            (0, 0, 0),
        )

        canvas.paste(
            resized,
            (
                (self.size - resized.width) // 2,
                (self.size - resized.height) // 2,
            ),
        )
        return canvas


tf_eff = T.Compose(
    [
        LetterboxResize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

tf_hc = T.Compose(
    [
        LetterboxResize(IMG_SIZE),
        T.ToTensor(),
    ]
)


def extract_hog(tensor_img: torch.Tensor) -> np.ndarray:
    img_np = (
        tensor_img.permute(1, 2, 0).detach().cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    feature = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return np.asarray(feature, dtype=np.float32)


def extract_hsv(tensor_img: torch.Tensor) -> np.ndarray:
    img_np = (
        tensor_img.permute(1, 2, 0).detach().cpu().numpy() * 255
    ).clip(0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    histogram_h = cv2.calcHist(
        [hsv], [0], None, [HSV_BINS[0]], [0, 180]
    ).flatten()
    histogram_s = cv2.calcHist(
        [hsv], [1], None, [HSV_BINS[1]], [0, 256]
    ).flatten()
    histogram_v = cv2.calcHist(
        [hsv], [2], None, [HSV_BINS[2]], [0, 256]
    ).flatten()

    histogram = np.concatenate(
        [histogram_h, histogram_s, histogram_v]
    ).astype(np.float32)

    total = float(histogram.sum())
    if total > 0:
        histogram /= total

    return histogram


# =============================================================================
# MODEL
# =============================================================================

class HybridLogoModel(nn.Module):
    def __init__(self, hog_dim: int, embed_dim: int = EMBED_DIM):
        super().__init__()

        efficientnet = models.efficientnet_v2_s(weights=None)

        self.backbone = nn.Sequential(
            *list(efficientnet.children())[:-1]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.hog_proj = nn.Sequential(
            nn.Linear(hog_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.embedding_layer = nn.Sequential(
            nn.Linear(1280 + 256, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),           # ← SiLU bukan ReLU
            nn.Linear(512, embed_dim),  # ← tanpa Dropout
        )

    def forward(
        self,
        img: torch.Tensor,
        hog_vec: torch.Tensor,
    ) -> torch.Tensor:
        deep_feature = self.pool(
            self.backbone(img)
        ).flatten(1)

        hog_feature = self.hog_proj(hog_vec)

        embedding = self.embedding_layer(
            torch.cat([deep_feature, hog_feature], dim=1)
        )

        return nn.functional.normalize(
            embedding,
            p=2,
            dim=1,
        )


class ModelWrapperCAM(nn.Module):
    def __init__(
        self,
        model: HybridLogoModel,
        hog_fixed: torch.Tensor,
        ref_emb: np.ndarray | None = None,
    ):
        super().__init__()
        self.model = model
        self.hog_fixed = hog_fixed

        if ref_emb is None:
            self.ref = None
        else:
            self.ref = torch.as_tensor(
                ref_emb,
                dtype=torch.float32,
            )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        embedding = self.model(img, self.hog_fixed)

        if self.ref is not None:
            reference = self.ref.to(embedding.device)
            return (
                embedding * reference
            ).sum(dim=1, keepdim=True)

        return embedding.norm(dim=1, keepdim=True)


# =============================================================================
# UTILITAS DATA
# =============================================================================

def l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)

    if array.ndim != 2:
        raise ValueError(
            f"Array harus 2D, ditemukan shape {array.shape}."
        )

    norm = np.linalg.norm(array, axis=1, keepdims=True)

    if np.any(norm <= 1e-12):
        count = int(np.sum(norm <= 1e-12))
        raise ValueError(
            f"Ditemukan {count} vektor database dengan norma nol."
        )

    return array / norm


def l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise ValueError("Vektor query memiliki norma nol.")

    return vector / norm


def build_unique_filename_index(
    root: Path,
) -> tuple[dict[str, Path], set[str]]:
    """Menyimpan hanya filename yang unik.

    Filename yang muncul lebih dari satu kali tidak digunakan sebagai fallback,
    agar aplikasi tidak menampilkan gambar yang salah.
    """

    if not root.is_dir():
        return {}, set()

    first_path: dict[str, Path] = {}
    duplicate_names: set[str] = set()

    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            filename = path.name

            if filename in first_path:
                duplicate_names.add(filename)
            else:
                first_path[filename] = path

    for filename in duplicate_names:
        first_path.pop(filename, None)

    return first_path, duplicate_names


def extract_relative_pdki_path(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip()
    if not normalized:
        return ""

    lowered = normalized.lower()

    marker = "/pdki/"
    marker_index = lowered.find(marker)

    if marker_index >= 0:
        return normalized[
            marker_index + len(marker):
        ].lstrip("/")

    if lowered.startswith("pdki/"):
        return normalized[len("pdki/"):].lstrip("/")

    return ""


def resolve_image_path(
    metadata: dict[str, Any],
    unique_filename_index: dict[str, Path],
) -> Path | None:
    """Resolusi path tanpa memilih file ambigu berdasarkan basename."""

    raw_path = str(metadata.get("path") or "").strip()

    if raw_path:
        original = Path(raw_path).expanduser()

        if original.is_file():
            return original

        relative_pdki = extract_relative_pdki_path(
            raw_path
        )

        if relative_pdki:
            candidate = (
                IMAGES_DIR / relative_pdki
            ).resolve()

            if candidate.is_file():
                return candidate

    for key in (
        "relative_path",
        "image_relative_path",
        "image_path",
        "filepath",
    ):
        relative_value = str(
            metadata.get(key) or ""
        ).strip()

        if not relative_value:
            continue

        normalized = relative_value.replace("\\", "/")
        relative_pdki = (
            extract_relative_pdki_path(normalized)
            or normalized.lstrip("/")
        )

        candidate = (
            IMAGES_DIR / relative_pdki
        ).resolve()

        if candidate.is_file():
            return candidate

    filename = Path(
        str(metadata.get("filename") or "").strip()
    ).name

    if filename:
        # Fallback hanya dilakukan bila filename unik.
        return unique_filename_index.get(filename)

    return None


def parse_nice_classes(value: Any) -> set[int]:
    if value is None:
        return set()

    if isinstance(value, (list, tuple, set)):
        result: set[int] = set()
        for item in value:
            result.update(parse_nice_classes(item))
        return result

    text = str(value).strip()
    if not text:
        return set()

    result = set()

    for token in re.findall(
        r"\d+(?:\.0+)?",
        text,
    ):
        try:
            class_number = int(float(token))
        except ValueError:
            continue

        if 1 <= class_number <= 45:
            result.add(class_number)

    return result


def safe_open_uploaded_logo(
    uploaded_file: Any,
) -> Image.Image:
    data = uploaded_file.getvalue()

    if not data:
        raise ValueError("File upload kosong.")

    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(
            "Ukuran file melebihi batas 10 MB."
        )

    try:
        with Image.open(io.BytesIO(data)) as source:
            source.load()
            image = ImageOps.exif_transpose(
                source
            ).copy()
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ValueError(
            "File tidak dapat dibaca sebagai gambar yang valid."
        ) from exc

    width, height = image.size

    if width <= 0 or height <= 0:
        raise ValueError("Dimensi gambar tidak valid.")

    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            "Resolusi gambar terlalu besar. "
            "Maksimum 25 juta piksel."
        )

    # Konsisten dengan letterbox hitam yang digunakan model.
    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        background = Image.new(
            "RGBA",
            rgba.size,
            (0, 0, 0, 255),
        )
        image = Image.alpha_composite(
            background,
            rgba,
        ).convert("RGB")
    else:
        image = image.convert("RGB")

    return image


# =============================================================================
# LOAD MODEL DAN DATABASE
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_model_and_db():
    required_files = [
        MODEL_PATH,
        EMBEDS_PATH,
        HSV_PATH,
        METADATA_PATH,
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "File berikut tidak ditemukan:\n- "
            + "\n- ".join(missing)
        )

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder gambar tidak ditemukan: {IMAGES_DIR}"
        )

    sample_tensor = tf_hc(
        Image.new("RGB", (100, 100), (0, 0, 0))
    )
    computed_hog_dim = len(
        extract_hog(sample_tensor)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Checkpoint harus berupa dictionary PyTorch."
        )

    checkpoint_hog_dim = int(
        checkpoint.get(
            "hog_dim",
            computed_hog_dim,
        )
    )

    checkpoint_embed_dim = int(
        checkpoint.get(
            "embed_dim",
            EMBED_DIM,
        )
    )

    if checkpoint_hog_dim != computed_hog_dim:
        raise ValueError(
            "Dimensi HOG tidak cocok. "
            f"Checkpoint={checkpoint_hog_dim}, "
            f"aplikasi={computed_hog_dim}."
        )

    if checkpoint_embed_dim != EMBED_DIM:
        raise ValueError(
            "Dimensi embedding checkpoint tidak cocok. "
            f"Checkpoint={checkpoint_embed_dim}, "
            f"aplikasi={EMBED_DIM}."
        )

    state_dict = checkpoint.get("model_state")

    if not isinstance(state_dict, dict):
        raise KeyError(
            "Checkpoint tidak memiliki key 'model_state'."
        )

    model = HybridLogoModel(
        hog_dim=computed_hog_dim,
        embed_dim=checkpoint_embed_dim,
    ).to(DEVICE)

    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval()

    db_embeds_raw = np.load(
        EMBEDS_PATH,
        allow_pickle=False,
    )
    db_hsv_raw = np.load(
        HSV_PATH,
        allow_pickle=False,
    )

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        db_meta = json.load(handle)

    if not isinstance(db_meta, list):
        raise TypeError(
            "pdki_metadata.json harus berupa JSON list."
        )

    if db_embeds_raw.ndim != 2:
        raise ValueError(
            "Embedding database harus 2D."
        )

    if db_hsv_raw.ndim != 2:
        raise ValueError(
            "Fitur HSV database harus 2D."
        )

    if db_embeds_raw.shape[1] != checkpoint_embed_dim:
        raise ValueError(
            "Dimensi embedding database tidak cocok. "
            f"Ditemukan {db_embeds_raw.shape[1]}, "
            f"diharapkan {checkpoint_embed_dim}."
        )

    if db_hsv_raw.shape[1] != HSV_DIM:
        raise ValueError(
            "Dimensi fitur HSV database tidak cocok. "
            f"Ditemukan {db_hsv_raw.shape[1]}, "
            f"diharapkan {HSV_DIM}."
        )

    if not (
        len(db_meta)
        == db_embeds_raw.shape[0]
        == db_hsv_raw.shape[0]
    ):
        raise ValueError(
            "Jumlah metadata, embedding, dan HSV "
            "tidak konsisten."
        )

    if not np.isfinite(
        db_embeds_raw
    ).all():
        raise ValueError(
            "Embedding database mengandung NaN/Inf."
        )

    if not np.isfinite(
        db_hsv_raw
    ).all():
        raise ValueError(
            "Fitur HSV database mengandung NaN/Inf."
        )

    db_embeds = l2_normalize_rows(
        db_embeds_raw
    )
    db_hsv = l2_normalize_rows(
        db_hsv_raw
    )

    unique_filename_index, duplicate_filenames = (
        build_unique_filename_index(IMAGES_DIR)
    )

    database_info = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "hog_dim": checkpoint_hog_dim,
        "embed_dim": checkpoint_embed_dim,
        "duplicate_filename_count": len(
            duplicate_filenames
        ),
        "unique_filename_fallback_count": len(
            unique_filename_index
        ),
    }

    return (
        model,
        db_embeds,
        db_hsv,
        db_meta,
        unique_filename_index,
        database_info,
    )


# =============================================================================
# INFERENCE DAN RETRIEVAL
# =============================================================================

@torch.no_grad()
def get_embedding(
    img_pil: Image.Image,
    model: HybridLogoModel,
):
    tensor_eff = tf_eff(
        img_pil
    ).unsqueeze(0).to(DEVICE)

    tensor_hc = tf_hc(
        img_pil
    )

    hog_vector = torch.as_tensor(
        extract_hog(tensor_hc),
        dtype=torch.float32,
        device=DEVICE,
    ).unsqueeze(0)

    hsv_vector = extract_hsv(
        tensor_hc
    )

    embedding = model(
        tensor_eff,
        hog_vector,
    ).detach().cpu().numpy().reshape(-1)

    embedding = l2_normalize_vector(
        embedding
    )
    hsv_vector = l2_normalize_vector(
        hsv_vector
    )

    return (
        embedding,
        hsv_vector,
        tensor_eff,
        tensor_hc,
        hog_vector,
    )


def retrieve(
    embedding_query: np.ndarray,
    hsv_query: np.ndarray,
    db_embeddings: np.ndarray,
    db_hsv: np.ndarray,
    *,
    k: int,
    alpha: float,
    threshold: float,
):
    if db_embeddings.shape[0] == 0:
        raise ValueError(
            "Database hasil filter kosong."
        )

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Alpha harus berada pada rentang 0–1."
        )

    effective_k = min(
        max(1, int(k)),
        db_embeddings.shape[0],
    )

    hybrid_scores = (
        db_embeddings @ embedding_query
    ).astype(np.float32)

    color_scores = (
        db_hsv @ hsv_query
    ).astype(np.float32)

    final_scores = (
        alpha * hybrid_scores
        + (1.0 - alpha) * color_scores
    ).astype(np.float32)

    # Tidak ada penghapusan otomatis berdasarkan skor.
    # Upload eksternal yang identik dengan merek terdaftar tetap harus muncul.
    top_indices = np.argpartition(
        final_scores,
        -effective_k,
    )[-effective_k:]

    top_indices = top_indices[
        np.argsort(
            final_scores[top_indices]
        )[::-1]
    ]

    top_final_scores = final_scores[
        top_indices
    ]
    top_hybrid_scores = hybrid_scores[
        top_indices
    ]
    top_color_scores = color_scores[
        top_indices
    ]

    above_threshold = (
        top_final_scores >= threshold
    )

    return (
        top_indices,
        top_final_scores,
        top_hybrid_scores,
        top_color_scores,
        above_threshold,
    )



# =============================================================================
# OCCLUSION SENSITIVITY
# =============================================================================

def _sliding_positions(
    length: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError(
            "Patch size dan stride harus lebih besar dari nol."
        )

    patch_size = min(patch_size, length)
    last_position = length - patch_size

    positions = list(
        range(0, last_position + 1, stride)
    )

    if not positions:
        positions = [0]
    elif positions[-1] != last_position:
        positions.append(last_position)

    return positions


def _make_occluded_images(
    image: Image.Image,
    patch_size: int,
    stride: int,
) -> tuple[
    np.ndarray,
    list[Image.Image],
    list[tuple[int, int, int, int]],
]:
    """Membuat variasi query dengan area lokal ditutup secara halus.

    Fill color diambil dari median border gambar. Feathered mask dipakai agar
    batas patch tidak menciptakan edge artifisial yang terlalu kuat.
    """

    letterboxed = LetterboxResize(
        IMG_SIZE
    )(image)

    base = np.asarray(
        letterboxed,
        dtype=np.float32,
    )

    border_width = max(
        4,
        IMG_SIZE // 28,
    )

    border_pixels = np.concatenate(
        [
            base[:border_width, :, :].reshape(-1, 3),
            base[-border_width:, :, :].reshape(-1, 3),
            base[:, :border_width, :].reshape(-1, 3),
            base[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    )

    fill_color = np.median(
        border_pixels,
        axis=0,
    ).astype(np.float32)

    x_positions = _sliding_positions(
        IMG_SIZE,
        patch_size,
        stride,
    )
    y_positions = _sliding_positions(
        IMG_SIZE,
        patch_size,
        stride,
    )

    images: list[Image.Image] = []
    coordinates: list[
        tuple[int, int, int, int]
    ] = []

    feather_sigma = max(
        2.0,
        patch_size / 12.0,
    )

    for y0 in y_positions:
        for x0 in x_positions:
            x1 = min(
                IMG_SIZE,
                x0 + patch_size,
            )
            y1 = min(
                IMG_SIZE,
                y0 + patch_size,
            )

            mask = np.zeros(
                (IMG_SIZE, IMG_SIZE),
                dtype=np.float32,
            )

            mask[
                y0:y1,
                x0:x1,
            ] = 1.0

            mask = cv2.GaussianBlur(
                mask,
                ksize=(0, 0),
                sigmaX=feather_sigma,
                sigmaY=feather_sigma,
            )

            mask = np.clip(
                mask,
                0.0,
                1.0,
            )[..., None]

            occluded = (
                base * (1.0 - mask)
                + fill_color.reshape(1, 1, 3) * mask
            )

            images.append(
                Image.fromarray(
                    np.clip(
                        occluded,
                        0,
                        255,
                    ).astype(np.uint8),
                    mode="RGB",
                )
            )

            coordinates.append(
                (x0, y0, x1, y1)
            )

    return (
        base.astype(np.uint8),
        images,
        coordinates,
    )


@torch.no_grad()
def _extract_features_for_images(
    images: list[Image.Image],
    model: HybridLogoModel,
    *,
    batch_size: int = DEFAULT_OCCLUSION_BATCH_SIZE,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    if not images:
        raise ValueError(
            "Daftar gambar occlusion kosong."
        )

    effective_tensors: list[torch.Tensor] = []
    hog_vectors: list[np.ndarray] = []
    hsv_vectors: list[np.ndarray] = []

    total = len(images)

    for index, image in enumerate(images):
        tensor_eff_item = tf_eff(image)
        tensor_hc_item = tf_hc(image)

        effective_tensors.append(
            tensor_eff_item
        )
        hog_vectors.append(
            extract_hog(
                tensor_hc_item
            )
        )
        hsv_vectors.append(
            l2_normalize_vector(
                extract_hsv(
                    tensor_hc_item
                )
            )
        )

        if progress_callback is not None:
            progress_callback(
                0.35
                * (index + 1)
                / total
            )

    all_embeddings: list[np.ndarray] = []

    for start in range(
        0,
        total,
        batch_size,
    ):
        end = min(
            total,
            start + batch_size,
        )

        image_batch = torch.stack(
            effective_tensors[start:end],
            dim=0,
        ).to(DEVICE)

        hog_batch = torch.as_tensor(
            np.stack(
                hog_vectors[start:end],
                axis=0,
            ),
            dtype=torch.float32,
            device=DEVICE,
        )

        embedding_batch = model(
            image_batch,
            hog_batch,
        ).detach().cpu().numpy()

        all_embeddings.append(
            embedding_batch.astype(
                np.float32
            )
        )

        if progress_callback is not None:
            progress_callback(
                0.35
                + 0.55
                * end
                / total
            )

    embeddings = l2_normalize_rows(
        np.concatenate(
            all_embeddings,
            axis=0,
        )
    )

    hsv_features = l2_normalize_rows(
        np.stack(
            hsv_vectors,
            axis=0,
        )
    )

    if progress_callback is not None:
        progress_callback(0.92)

    return embeddings, hsv_features


def _build_sensitivity_map(
    score_drops: np.ndarray,
    coordinates: list[
        tuple[int, int, int, int]
    ],
) -> tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.float32,
    )
    counts = np.zeros(
        (IMG_SIZE, IMG_SIZE),
        dtype=np.float32,
    )

    positive_drops = np.maximum(
        np.asarray(
            score_drops,
            dtype=np.float32,
        ),
        0.0,
    )

    for drop, (
        x0,
        y0,
        x1,
        y1,
    ) in zip(
        positive_drops,
        coordinates,
    ):
        canvas[y0:y1, x0:x1] += float(
            drop
        )
        counts[y0:y1, x0:x1] += 1.0

    raw_map = np.divide(
        canvas,
        counts,
        out=np.zeros_like(canvas),
        where=counts > 0,
    )

    maximum = float(
        raw_map.max()
    )

    if maximum > 1e-12:
        normalized = raw_map / maximum
    else:
        normalized = np.zeros_like(
            raw_map
        )

    return (
        raw_map,
        normalized.astype(
            np.float32
        ),
    )


def _overlay_sensitivity(
    base_rgb: np.ndarray,
    normalized_map: np.ndarray,
) -> np.ndarray:
    base_float = np.asarray(
        base_rgb,
        dtype=np.float32,
    )

    heatmap_rgb = heatmap_to_rgb(
        normalized_map
    ).astype(np.float32)

    strength = np.clip(
        normalized_map,
        0.0,
        1.0,
    )[..., None]

    alpha_map = (
        0.15
        + 0.55 * strength
    )

    overlay = (
        base_float * (1.0 - alpha_map)
        + heatmap_rgb * alpha_map
    )

    return np.clip(
        overlay,
        0,
        255,
    ).astype(np.uint8)


def _draw_influential_box(
    base_rgb: np.ndarray,
    box: tuple[int, int, int, int],
) -> np.ndarray:
    output = np.asarray(
        base_rgb,
        dtype=np.uint8,
    ).copy()

    x0, y0, x1, y1 = box

    cv2.rectangle(
        output,
        (x0, y0),
        (max(x0, x1 - 1), max(y0, y1 - 1)),
        (255, 0, 0),
        3,
    )

    return output


def compute_occlusion_sensitivity(
    image_query: Image.Image,
    model: HybridLogoModel,
    *,
    reference_embedding: np.ndarray,
    reference_hsv: np.ndarray,
    baseline_embedding_score: float,
    baseline_hsv_score: float,
    alpha: float,
    patch_size: int,
    stride: int,
    batch_size: int = DEFAULT_OCCLUSION_BATCH_SIZE,
    progress_callback=None,
) -> dict[str, Any]:
    """Mengukur perubahan skor ketika area query ditutup.

    Peta utama menjelaskan skor fusion akhir. Peta embedding dan HSV juga
    disediakan agar pemeriksa dapat melihat sumber kontribusi secara terpisah.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "Alpha harus berada pada rentang 0–1."
        )

    reference_embedding = l2_normalize_vector(
        reference_embedding
    )
    reference_hsv = l2_normalize_vector(
        reference_hsv
    )

    (
        base_rgb,
        occluded_images,
        coordinates,
    ) = _make_occluded_images(
        image_query,
        patch_size=patch_size,
        stride=stride,
    )

    (
        occluded_embeddings,
        occluded_hsv,
    ) = _extract_features_for_images(
        occluded_images,
        model,
        batch_size=batch_size,
        progress_callback=progress_callback,
    )

    occluded_embedding_scores = (
        occluded_embeddings
        @ reference_embedding
    ).astype(np.float32)

    occluded_hsv_scores = (
        occluded_hsv
        @ reference_hsv
    ).astype(np.float32)

    baseline_final_score = float(
        alpha * baseline_embedding_score
        + (1.0 - alpha) * baseline_hsv_score
    )

    occluded_final_scores = (
        alpha * occluded_embedding_scores
        + (1.0 - alpha) * occluded_hsv_scores
    ).astype(np.float32)

    embedding_drops = (
        float(baseline_embedding_score)
        - occluded_embedding_scores
    )
    hsv_drops = (
        float(baseline_hsv_score)
        - occluded_hsv_scores
    )
    final_drops = (
        baseline_final_score
        - occluded_final_scores
    )

    component_data = {}

    for name, drops in (
        ("final", final_drops),
        ("embedding", embedding_drops),
        ("hsv", hsv_drops),
    ):
        raw_map, normalized_map = (
            _build_sensitivity_map(
                drops,
                coordinates,
            )
        )

        component_data[name] = {
            "raw_map": raw_map,
            "normalized_map": normalized_map,
            "heatmap_rgb": heatmap_to_rgb(
                normalized_map
            ),
            "overlay_rgb": _overlay_sensitivity(
                base_rgb,
                normalized_map,
            ),
            "max_positive_drop": float(
                np.maximum(
                    drops,
                    0.0,
                ).max()
            ),
        }

    strongest_index = int(
        np.argmax(final_drops)
    )
    strongest_box = coordinates[
        strongest_index
    ]

    if progress_callback is not None:
        progress_callback(1.0)

    return {
        "base_rgb": base_rgb,
        "boxed_rgb": _draw_influential_box(
            base_rgb,
            strongest_box,
        ),
        "coordinates": coordinates,
        "patch_count": len(coordinates),
        "strongest_box": strongest_box,
        "baseline_embedding_score": float(
            baseline_embedding_score
        ),
        "baseline_hsv_score": float(
            baseline_hsv_score
        ),
        "baseline_final_score": baseline_final_score,
        "minimum_embedding_score": float(
            occluded_embedding_scores.min()
        ),
        "minimum_hsv_score": float(
            occluded_hsv_scores.min()
        ),
        "minimum_final_score": float(
            occluded_final_scores.min()
        ),
        "maximum_final_drop": float(
            np.maximum(
                final_drops,
                0.0,
            ).max()
        ),
        "components": component_data,
    }


def _occlusion_cache_key(
    uploaded_bytes: bytes,
    rank1_metadata: dict[str, Any],
    *,
    alpha: float,
    patch_size: int,
    stride: int,
) -> str:
    identifier = "|".join(
        [
            safe_text(
                rank1_metadata.get(
                    "application_id"
                )
            ),
            safe_text(
                rank1_metadata.get(
                    "nomor_permohonan"
                )
            ),
            safe_text(
                rank1_metadata.get(
                    "path"
                )
            ),
            f"{alpha:.6f}",
            str(patch_size),
            str(stride),
            APP_VERSION,
        ]
    )

    digest = hashlib.sha256()
    digest.update(uploaded_bytes)
    digest.update(
        identifier.encode(
            "utf-8",
            errors="replace",
        )
    )
    return digest.hexdigest()


# =============================================================================
# EIGENGRADCAM
# =============================================================================

def get_cam_target_layer(
    model: HybridLogoModel,
) -> nn.Module:
    return model.backbone[0][-1]


def generate_heatmap(
    model: HybridLogoModel,
    hog_vector: torch.Tensor,
    tensor_eff: torch.Tensor,
    ref_embedding: np.ndarray | None = None,
):
    wrapped = ModelWrapperCAM(
        model=model,
        hog_fixed=hog_vector,
        ref_emb=ref_embedding,
    )
    wrapped.eval()

    target_layer = get_cam_target_layer(
        model
    )

    with torch.enable_grad():
        with EigenGradCAM(
            model=wrapped,
            target_layers=[target_layer],
        ) as cam:
            grayscale = cam(
                input_tensor=tensor_eff,
                targets=None,
            )[0]

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    img_np = (
        tensor_eff.squeeze(0)
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
        * std
        + mean
    ).clip(0, 1).astype(np.float32)

    if float(img_np.mean()) < 0.2:
        img_display = np.clip(
            img_np * 3.0,
            0,
            1,
        )
    else:
        img_display = img_np

    overlay = show_cam_on_image(
        img_display,
        grayscale,
        use_rgb=True,
    )

    return overlay, grayscale


def heatmap_to_rgb(
    grayscale: np.ndarray,
) -> np.ndarray:
    colored = cv2.applyColorMap(
        np.uint8(
            255 * np.clip(grayscale, 0, 1)
        ),
        cv2.COLORMAP_JET,
    )

    return cv2.cvtColor(
        colored,
        cv2.COLOR_BGR2RGB,
    )


# =============================================================================
# PRESENTASI
# =============================================================================

def status_badge(status: Any):
    text = str(
        status or "N/A"
    ).strip()
    lowered = text.lower()

    if "didaftar" in lowered:
        st.success(f"✅ {text}")
    elif "tolak" in lowered:
        st.error(f"🚫 {text}")
    elif any(
        token in lowered
        for token in (
            "pengumuman",
            "substantif",
            "proses",
        )
    ):
        st.warning(f"⏳ {text}")
    else:
        st.info(f"ℹ️ {text}")


def safe_text(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "null",
        "nat",
    }:
        return default

    return text


def build_csv_report(
    rows: list[dict[str, Any]],
) -> str:
    buffer = io.StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "rank",
            "score_final",
            "score_embedding",
            "score_hsv",
            "di_atas_threshold",
            "nama_merek",
            "pemilik",
            "nomor_permohonan",
            "kelas_nice",
            "tanggal_permohonan",
            "status_permohonan",
        ],
    )

    writer.writeheader()

    for row in rows:
        writer.writerow(
            {
                "rank": row["rank"],
                "score_final": (
                    f"{row['score_final']:.6f}"
                ),
                "score_embedding": (
                    f"{row['score_embedding']:.6f}"
                ),
                "score_hsv": (
                    f"{row['score_hsv']:.6f}"
                ),
                "di_atas_threshold": (
                    "Ya"
                    if row["above_threshold"]
                    else "Tidak"
                ),
                "nama_merek": row["nama_merek"],
                "pemilik": row["owner"],
                "nomor_permohonan": (
                    row["nomor_permohonan"]
                ),
                "kelas_nice": row["kelas_nice"],
                "tanggal_permohonan": (
                    row["tanggal_permohonan"]
                ),
                "status_permohonan": row["status"],
            }
        )

    return buffer.getvalue()


# =============================================================================
# APLIKASI
# =============================================================================

def inject_djki_ui():
    """Menerapkan styling UI berbasis palette DJKI."""
    st.markdown(
        f"""
        <style>
        /* ================================================================
           GLOBAL
        ================================================================= */
        .stApp {{
            background-color: {COLOR_BACKGROUND};
            color: {COLOR_TEXT};
        }}

        .main .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }}

        /* ================================================================
           SIDEBAR
        ================================================================= */
        [data-testid="stSidebar"] {{
            background: linear-gradient(
                180deg,
                {COLOR_NAVY} 0%,
                #25396F 100%
            );
        }}

        [data-testid="stSidebar"] * {{
            color: white;
        }}

        [data-testid="stSidebar"] hr {{
            border-color: rgba(255,255,255,0.25);
        }}

        [data-testid="stSidebar"] .stCaption {{
            color: rgba(255,255,255,0.78);
        }}

        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {{
            background-color: rgba(255,255,255,0.96);
            color: {COLOR_TEXT};
            border-radius: 8px;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background-color: rgba(255,255,255,0.96);
            color: {COLOR_TEXT};
        }}

        /* ================================================================
           TYPOGRAPHY
        ================================================================= */
        h1 {{
            color: {COLOR_NAVY};
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        h2 {{
            color: {COLOR_NAVY};
            font-weight: 750;
        }}

        h3 {{
            color: {COLOR_BLUE};
            font-weight: 700;
        }}

        p, label, .stMarkdown {{
            color: {COLOR_TEXT};
        }}

        /* ================================================================
           BUTTONS
        ================================================================= */
        .stButton > button {{
            background-color: {COLOR_BLUE};
            color: white;
            border: 1px solid {COLOR_BLUE};
            border-radius: 8px;
            font-weight: 700;
            min-height: 2.6rem;
            transition: all 0.2s ease;
        }}

        .stButton > button:hover {{
            background-color: {COLOR_NAVY};
            border-color: {COLOR_NAVY};
            color: white;
            transform: translateY(-1px);
        }}

        .stDownloadButton > button {{
            border: 1px solid {COLOR_BLUE};
            color: {COLOR_NAVY};
            background-color: white;
            border-radius: 8px;
            font-weight: 600;
        }}
        
        .stDownloadButton > button:hover {{
            background-color: {COLOR_BLUE} !important;
            color: {COLOR_WHITE} !important;
            border-color: {COLOR_BLUE} !important;
        }}
        
        .stDownloadButton > button:hover *,
        .stDownloadButton > button:hover p,
        .stDownloadButton > button:hover span,
        .stDownloadButton > button:hover label {{
            color: {COLOR_WHITE} !important;
        }}
        
        /* Hanya primary button yang tidak disabled */
        .stButton > button[kind="primary"]:not(:disabled) p {{
            color: {COLOR_WHITE} !important;
        }}
        /* ================================================================
           METRICS
        ================================================================= */
        [data-testid="stMetric"] {{
            background-color: {COLOR_WHITE};
            border-top: 5px solid {COLOR_YELLOW};
            border-radius: 10px;
            padding: 14px 16px;
            box-shadow: 0 2px 10px rgba(29,45,92,0.10);
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLOR_MUTED};
            font-weight: 600;
        }}

        [data-testid="stMetricValue"] {{
            color: {COLOR_NAVY};
            font-weight: 800;
        }}

        /* ================================================================
           FILE UPLOADER
        ================================================================= */
        [data-testid="stFileUploader"] {{
            background-color: white !important;
            border: 2px dashed {COLOR_BLUE} !important;
            border-radius: 12px !important;
            padding: 6px !important;
        }}
        
        [data-testid="stFileUploaderDropzone"] {{
            background: transparent !important;
        }}
        
        /* Center isi file uploader */
        [data-testid="stFileUploader"] section {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 12px !important;
            text-align: center !important;
        }}
        
        [data-testid="stFileUploader"] section > div {{
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }}
        
        [data-testid="stFileUploader"] label {{
            text-align: center !important;
            font-size: 0.95rem !important;
            font-weight: 600 !important;
            color: {COLOR_NAVY} !important;
        }}
        
        [data-testid="stFileUploader"] section::before {{
            content: "Klik di sini atau seret gambar ke sini";
            display: block;
            text-align: center;
            font-size: 0.95rem;
            font-weight: 600;
            color: {COLOR_NAVY};
            margin-bottom: 8px;
        }}
        
        /* ================================================================
           CARDS / CONTAINERS
        ================================================================= */
        .djki-header {{
            background: linear-gradient(
                135deg,
                {COLOR_NAVY} 0%,
                {COLOR_BLUE} 100%
            );
            color: white;
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 18px;
            box-shadow: 0 4px 14px rgba(29,45,92,0.18);
        }}

        .djki-header h1,
        .djki-header p {{
            color: white !important;
        }}

        .djki-section-title {{
            color: {COLOR_NAVY};
            font-weight: 800;
            margin: 12px;
        }}

        /* ================================================================
           ALERTS
        ================================================================= */
        [data-testid="stAlert"] {{
            border-radius: 9px;
        }}

        /* ================================================================
           TABS
        ================================================================= */
        button[data-baseweb="tab"] {{
            color: {COLOR_NAVY};
            font-weight: 600;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {COLOR_BLUE};
        }}

        /* ================================================================
           SLIDERS
        ================================================================= */
        [data-testid="stSlider"] [role="slider"] {{
            background-color: {COLOR_BLUE};
        }}

        /* ================================================================
           EXPANDERS
        ================================================================= */
        [data-testid="stExpander"] {{
            background-color: white;
            border: 1px solid rgba(48,73,138,0.18);
            border-radius: 10px;
        }}
        
        [data-testid="stSidebar"] [data-testid="stExpander"] {{
            background-color: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.2) !important;
            border-radius: 8px !important;
        }}
        
        [data-testid="stSidebar"] details summary,
        [data-testid="stSidebar"] details[open] summary,
        [data-testid="stSidebar"] details summary:hover,
        [data-testid="stSidebar"] details summary:focus,
        [data-testid="stSidebar"] details summary:active {{
            background-color: rgba(255,255,255,0.08) !important;
            border-radius: 6px !important;
        }}
        
        [data-testid="stSidebar"] details summary *,
        [data-testid="stSidebar"] details[open] summary *,
        [data-testid="stSidebar"] details summary:hover *,
        [data-testid="stSidebar"] details summary:focus *,
        [data-testid="stSidebar"] details summary:active * {{
            color: White !important;
            fill: White !important;
            background-color: transparent !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stExpander"] p,
        [data-testid="stSidebar"] [data-testid="stExpander"] label,
        [data-testid="stSidebar"] [data-testid="stExpander"] span {{
            color: {COLOR_WHITE} !important;
        }}

        /* ================================================================
           Tooltip
        ================================================================= */        
        [data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg {{
            fill: {COLOR_WHITE} !important;
            color: {COLOR_WHITE} !important;
        }}
        [data-testid="stSidebar"] button[title="Show help text"] {{
            color: {COLOR_WHITE} !important;
        }}
        [data-testid="stSidebar"] .eyeqlp51 {{
            color: {COLOR_WHITE} !important;
        }}
        
        /* ================================================================
           RESULT ROW
        ================================================================= */
        .result-rank {{
            color: {COLOR_NAVY};
            font-weight: 800;
        }}

        .score-highlight {{
            color: {COLOR_BLUE};
            font-weight: 800;
        }}

        .threshold-warning {{
            color: {COLOR_RED_WARN};
            font-weight: 700;
        }}

        .threshold-safe {{
            color: #2E7D32;
            font-weight: 700;
        }}

        /* ================================================================
           DIVIDERS
        ================================================================= */
        hr {{
            border-color: rgba(29,45,92,0.18);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )



def main():
    st.set_page_config(
        page_title="DSS Kemiripan Logo Merek",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_djki_ui()

    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding: 10px 0 6px;">
            <span style="font-size:2rem;">⚖️</span><br>
            <span style="font-weight:800; font-size:1.1rem; color:white;">DSS Kemiripan Logo</span><br>
            <span style="font-size:0.78rem; opacity:0.78;">Pemeriksaan Substantif Merek | DJKI</span>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div style='text-align:center;font-size:0.72rem;opacity:0.6;'>v{APP_VERSION}</div>",
                    unsafe_allow_html=True)
        st.divider()

        st.markdown("### ⚙️ Pengaturan")

        top_k = st.slider(
            "Jumlah kandidat (Top-K)",
            min_value=3,
            max_value=20,
            value=DEFAULT_TOP_K,
        )

        alpha = st.slider(
            "Bobot embedding (α)",
            min_value=0.0,
            max_value=1.0,
            value=float(
                np.clip(
                    DEFAULT_ALPHA_FUSION,
                    0.0,
                    1.0,
                )
            ),
            step=0.05,
            help=(
                "Skor akhir = α × skor CNN-HOG "
                "+ (1−α) × skor HSV. "
                "Nilai final sebaiknya dikalibrasi "
                "menggunakan validation set."
            ),
        )

        threshold = st.slider(
            "Threshold kemiripan operasional",
            min_value=0.0,
            max_value=1.0,
            value=float(
                np.clip(
                    DEFAULT_SIMILARITY_THRESHOLD,
                    0.0,
                    1.0,
                )
            ),
            step=0.01,
            help=(
                "Threshold ini merupakan parameter DSS, "
                "bukan batas hukum. Nilai final harus "
                "ditentukan melalui evaluasi penelitian."
            ),
        )

        filter_kelas = st.text_input(
            "Filter Kelas Nice",
            placeholder=(
                "contoh: 25 atau 1 "
                "(kosongkan = semua)"
            ),
        )

        explainability_method = st.selectbox(
            "Metode Explainability",
            options=[
                "Occlusion Sensitivity (disarankan)",
                "Occlusion + EigenGradCAM",
                "EigenGradCAM saja",
                "Tidak ditampilkan",
            ],
            index=0,
            help=(
                "Occlusion Sensitivity menjelaskan pengaruh area gambar "
                "terhadap skor fusion akhir. EigenGradCAM hanya "
                "menjelaskan cabang CNN secara kualitatif."
            ),
        )

        occlusion_patch_size = DEFAULT_OCCLUSION_PATCH_SIZE
        occlusion_stride = DEFAULT_OCCLUSION_STRIDE

        if "Occlusion" in explainability_method:
            with st.expander(
                "Pengaturan Occlusion Sensitivity",
                expanded=False,
            ):
                occlusion_patch_size = st.select_slider(
                    "Ukuran patch",
                    options=[48, 56, 64],
                    value=DEFAULT_OCCLUSION_PATCH_SIZE,
                    help=(
                        "Patch lebih kecil memberi detail lebih tinggi "
                        "tetapi membutuhkan waktu lebih lama."
                    ),
                )

                occlusion_stride = st.select_slider(
                    "Stride",
                    options=[24, 28, 32],
                    value=DEFAULT_OCCLUSION_STRIDE,
                    help=(
                        "Stride lebih kecil menghasilkan pemetaan lebih rapat "
                        "tetapi membutuhkan lebih banyak inferensi."
                    ),
                )

        show_component_scores = st.checkbox(
            "Tampilkan komponen skor",
            value=True,
        )

        st.divider()
        st.info(
            "Sistem ini merupakan alat bantu DSS "
            "dan **bukan penentu keputusan akhir**.",
            icon="ℹ️",
        )
        st.caption(
            "Ghania Fazila (41122100060) | "
            "Universitas Widyatama"
        )

    st.markdown(
        """
        <div class="djki-header">
            <h1>Sistem Analisis Kemiripan Logo Merek Dagang</h1>
            <p>
                Decision Support System untuk Pemeriksaan Substantif Merek
                | DJKI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner(
        "Memuat model dan database PDKI..."
    ):
        try:
            (
                model,
                db_embeddings,
                db_hsv,
                db_metadata,
                unique_filename_index,
                database_info,
            ) = load_model_and_db()
        except Exception as exc:
            st.error(
                "Model/database gagal dimuat."
            )
            st.exception(exc)
            st.stop()

    st.success(
        f"Model siap. Database: "
        f"{len(db_metadata):,} logo PDKI.",
        icon="✅",
    )

    with st.expander(
        "Informasi teknis model/database",
        expanded=False,
    ):
        st.json(
            {
                "device": str(DEVICE),
                **database_info,
                "embedding_shape": list(
                    db_embeddings.shape
                ),
                "hsv_shape": list(
                    db_hsv.shape
                ),
                "alpha_default": (
                    DEFAULT_ALPHA_FUSION
                ),
                "threshold_default": (
                    DEFAULT_SIMILARITY_THRESHOLD
                ),
            }
        )

    st.markdown(
        f"<h3 style='text-align:center; color:{COLOR_NAVY};'>"
        f"Upload Logo Query</h3>",
        unsafe_allow_html=True
    )
    
    uploaded = st.file_uploader(
        "Klik di sini atau seret gambar ke sini",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )
    
    
    image_query: Image.Image | None = None

    if uploaded is not None:
        try:
            image_query = safe_open_uploaded_logo(uploaded)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(
                    image_query,
                    caption="Logo Query",
                    use_container_width=True,
                )
        except ValueError as exc:
            st.error(str(exc))

    analyze_clicked = st.button(
        "Analisis Kemiripan",
        type="primary",
        use_container_width=True,
        disabled=image_query is None,
    )

    if not analyze_clicked:
        return

    assert image_query is not None

    with st.spinner(
        "Mengekstrak fitur logo..."
    ):
        try:
            (
                embedding_query,
                hsv_query,
                tensor_eff,
                _tensor_hc,
                hog_vector,
            ) = get_embedding(
                image_query,
                model,
            )
        except Exception as exc:
            st.error(
                "Ekstraksi fitur query gagal."
            )
            st.exception(exc)
            st.stop()

    requested_classes = parse_nice_classes(
        filter_kelas
    )

    if filter_kelas.strip() and not requested_classes:
        st.warning(
            "Filter Kelas Nice tidak valid. "
            "Gunakan angka kelas 1–45."
        )
        st.stop()

    if requested_classes:
        valid_indices = [
            index
            for index, metadata in enumerate(
                db_metadata
            )
            if parse_nice_classes(
                metadata.get("kelas_nice")
            )
            & requested_classes
        ]

        if not valid_indices:
            requested_text = ", ".join(
                map(str, sorted(requested_classes))
            )
            st.warning(
                "Tidak ada logo untuk Kelas Nice "
                f"{requested_text} di database."
            )
            st.stop()

        index_map = np.asarray(
            valid_indices,
            dtype=np.int64,
        )

        filtered_embeddings = db_embeddings[
            index_map
        ]
        filtered_hsv = db_hsv[
            index_map
        ]
        filtered_metadata = [
            db_metadata[index]
            for index in valid_indices
        ]

        st.caption(
            "Filter aktif: Kelas Nice "
            + ", ".join(
                map(str, sorted(requested_classes))
            )
            + f" ({len(valid_indices):,} logo)"
        )
    else:
        filtered_embeddings = db_embeddings
        filtered_hsv = db_hsv
        filtered_metadata = db_metadata

    effective_k = min(
        top_k,
        len(filtered_metadata),
    )

    with st.spinner(
        "Mencari kandidat paling mirip..."
    ):
        try:
            (
                top_indices,
                top_final_scores,
                top_embedding_scores,
                top_hsv_scores,
                above_threshold,
            ) = retrieve(
                embedding_query,
                hsv_query,
                filtered_embeddings,
                filtered_hsv,
                k=effective_k,
                alpha=alpha,
                threshold=threshold,
            )
        except Exception as exc:
            st.error(
                "Proses retrieval gagal."
            )
            st.exception(exc)
            st.stop()

    above_count = int(
        above_threshold.sum()
    )

    rank1_score = float(
        top_final_scores[0]
    )

    if len(top_final_scores) >= 2:
        rank1_rank2_margin = float(
            top_final_scores[0]
            - top_final_scores[1]
        )
    else:
        rank1_rank2_margin = math.nan

    rank1_last_margin = float(
        top_final_scores[0]
        - top_final_scores[-1]
    )

    st.divider()
    st.markdown('<div class="djki-section-title">RINGKASAN HASIL</div>', unsafe_allow_html=True)

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Kandidat diperiksa",
        str(len(top_indices)),
    )

    metric_columns[1].metric(
        "Di atas threshold",
        str(above_count),
    )

    metric_columns[2].metric(
        "Skor tertinggi",
        f"{rank1_score:.4f}",
    )

    metric_columns[3].metric(
        "Margin Rank-1/Rank-2",
        (
            f"{rank1_rank2_margin:.4f}"
            if not math.isnan(
                rank1_rank2_margin
            )
            else "N/A"
        ),
    )

    st.caption(
        f"Rentang skor Top-{len(top_indices)}: "
        f"{top_final_scores[-1]:.4f}–"
        f"{top_final_scores[0]:.4f}; "
        f"margin Rank-1/Rank-{len(top_indices)}: "
        f"{rank1_last_margin:.4f}."
    )

    if above_count == 0:
        st.success(
            "Tidak ada kandidat pada hasil Top-K "
            "yang berada di atas threshold operasional. "
            "Pemeriksaan substantif tetap diperlukan."
        )
    elif above_count <= 2:
        st.warning(
            f"{above_count} kandidat berada di atas "
            "threshold operasional dan perlu diperiksa."
        )
    else:
        st.error(
            f"{above_count} kandidat berada di atas "
            "threshold operasional. Lakukan pemeriksaan "
            "lebih mendalam terhadap kandidat teratas."
        )

    st.info(
        f"Skor akhir menggunakan α={alpha:.2f}: "
        f"{alpha:.0%} embedding CNN-HOG dan "
        f"{1 - alpha:.0%} HSV.",
        icon="ℹ️",
    )

    st.divider()

    rank1_metadata = filtered_metadata[
        int(top_indices[0])
    ]

    rank1_path = resolve_image_path(
        rank1_metadata,
        unique_filename_index,
    )

    rank1_name = safe_text(
        rank1_metadata.get(
            "nama_merek"
        ),
        "Rank-1",
    )

    show_occlusion = (
        "Occlusion"
        in explainability_method
    )
    show_eigengradcam = (
        "EigenGradCAM"
        in explainability_method
    )

    if explainability_method != "Tidak ditampilkan":
        st.markdown('<div class="djki-section-title">EXPLAINABILITY</div>', unsafe_allow_html=True)

    if show_occlusion:
        st.markdown(
            "### Peta Sensitivitas Skor Kemiripan"
        )
        st.caption(
            "Area merah menunjukkan bagian logo query yang, ketika ditutup, "
            "paling menurunkan skor terhadap kandidat Rank-1. Metode ini "
            "mengukur skor fusion akhir sehingga mencakup CNN-HOG dan HSV."
        )

        uploaded_bytes = uploaded.getvalue()

        progress = st.progress(
            0.0,
            text=(
                "Menghitung Occlusion Sensitivity. "
                "Proses ini dapat memerlukan beberapa saat pada CPU..."
            ),
        )

        try:
            occlusion_result = (
                compute_occlusion_sensitivity(
                    image_query,
                    model,
                    reference_embedding=filtered_embeddings[
                        int(top_indices[0])
                    ],
                    reference_hsv=filtered_hsv[
                        int(top_indices[0])
                    ],
                    baseline_embedding_score=float(
                        top_embedding_scores[0]
                    ),
                    baseline_hsv_score=float(
                        top_hsv_scores[0]
                    ),
                    alpha=alpha,
                    patch_size=occlusion_patch_size,
                    stride=occlusion_stride,
                    batch_size=DEFAULT_OCCLUSION_BATCH_SIZE,
                    progress_callback=lambda value: (
                        progress.progress(
                            min(1.0, max(0.0, float(value))),
                            text="Menghitung Occlusion Sensitivity...",
                        )
                    ),
                )
            )
        except Exception as exc:
            progress.empty()
            st.warning("Occlusion Sensitivity tidak dapat dihitung.")
            st.exception(exc)
            occlusion_result = None
        else:
            progress.empty()

            if occlusion_result is not None:
    
                # Baris 1: Logo Query (3 panel) + Rank-1 (1 panel)
                col_h, col_o, col_b, col_r = st.columns(4)
    
                col_h.markdown("**Logo Query**")
                col_h.image(
                    occlusion_result["components"]["final"]["heatmap_rgb"],
                    caption="Heatmap Skor Akhir",
                    use_container_width=True,
                )
    
                col_o.markdown("&nbsp;", unsafe_allow_html=True)
                col_o.image(
                    occlusion_result["components"]["final"]["overlay_rgb"],
                    caption="Overlay pada Query",
                    use_container_width=True,
                )
    
                col_b.markdown("&nbsp;", unsafe_allow_html=True)
                col_b.image(
                    occlusion_result["boxed_rgb"],
                    caption="Area Paling Berpengaruh",
                    use_container_width=True,
                )
    
                col_r.markdown("**Kandidat Rank-1**")
                if rank1_path is not None:
                    try:
                        rank1_img = Image.open(rank1_path).convert("RGB")
                        canvas = Image.new("RGB", (300, 300), (255, 255, 255))
                        rank1_img.thumbnail((300, 300), Image.LANCZOS)
                        offset = (
                            (300 - rank1_img.width) // 2,
                            (300 - rank1_img.height) // 2,
                        )
                        canvas.paste(rank1_img, offset)
                        col_r.image(
                            canvas,
                            caption=rank1_name,
                            use_container_width=True,
                        )
                    except Exception:
                        col_r.warning("Gambar tidak dapat dimuat.")
                else:
                    col_r.warning("Path tidak ditemukan.")
                    
                x0, y0, x1, y1 = occlusion_result["strongest_box"]
                st.caption(
                    "Heatmap menunjukkan area pada logo query yang paling "
                    "berkontribusi terhadap kemiripan dengan kandidat Rank-1."
                )
                st.caption(
                    f"Penurunan skor maksimum: {occlusion_result['maximum_final_drop']:.4f} | "
                    f"Patch diuji: {occlusion_result['patch_count']} | "
                    f"Area berpengaruh: x={x0}–{x1}, y={y0}–{y1}"
                )
    
                tab_final, tab_emb, tab_hsv = st.tabs(
                    ["Skor Fusion Akhir", "CNN-HOG", "HSV"]
                )
                for tab, name, desc in [
                    (tab_final, "final", "Perubahan skor akhir setelah fusion CNN-HOG dan HSV."),
                    (tab_emb, "embedding", "Perubahan skor embedding CNN-HOG ketika area ditutup."),
                    (tab_hsv, "hsv", "Perubahan kemiripan warna HSV ketika area ditutup."),
                ]:
                    with tab:
                        st.caption(desc)
                        comp = occlusion_result["components"][name]
                        c1, c2, c3 = st.columns(3)
                        c1.image(
                            occlusion_result["base_rgb"],
                            caption="Query Asli",
                            use_container_width=True,
                        )
                        c2.image(
                            comp["heatmap_rgb"],
                            caption="Heatmap",
                            use_container_width=True,
                        )
                        c3.image(
                            comp["overlay_rgb"],
                            caption="Overlay",
                            use_container_width=True,
                        )
                        st.metric(
                            "Penurunan maksimum",
                            f"{comp['max_positive_drop']:.4f}"
                        )
            
            st.info(
                "Interpretasi: heatmap menunjukkan sensitivitas skor, "
                "bukan kesamaan hukum. Peta yang menyebar berarti model "
                "menggunakan bentuk/warna global; peta yang terpusat "
                "berarti area lokal tertentu memiliki pengaruh lebih besar.",
                icon="ℹ️",
            )

        st.divider()

    if show_eigengradcam:
        with st.expander(
            "Visualisasi tambahan — EigenGradCAM (cabang CNN)",
            expanded=not show_occlusion,
        ):
            st.caption(
                "EigenGradCAM hanya menjelaskan feature map cabang CNN. "
                "Visualisasi ini tidak menjelaskan kontribusi HOG, HSV, "
                "atau keseluruhan skor fusion secara langsung."
            )

            try:
                with st.spinner(
                    "Membuat EigenGradCAM..."
                ):
                    overlay_query, gray_query = (
                        generate_heatmap(
                            model=model,
                            hog_vector=hog_vector,
                            tensor_eff=tensor_eff,
                            ref_embedding=filtered_embeddings[
                                int(top_indices[0])
                            ],
                        )
                    )

                    rank1_image_cam = None
                    overlay_rank1 = None
                    gray_rank1 = None

                    if rank1_path is not None:
                        rank1_image_cam = (
                            Image.open(
                                rank1_path
                            ).convert("RGB")
                        )

                        tensor_eff_rank1 = tf_eff(
                            rank1_image_cam
                        ).unsqueeze(0).to(DEVICE)

                        tensor_hc_rank1 = tf_hc(
                            rank1_image_cam
                        )

                        hog_rank1 = torch.as_tensor(
                            extract_hog(
                                tensor_hc_rank1
                            ),
                            dtype=torch.float32,
                            device=DEVICE,
                        ).unsqueeze(0)

                        (
                            overlay_rank1,
                            gray_rank1,
                        ) = generate_heatmap(
                            model=model,
                            hog_vector=hog_rank1,
                            tensor_eff=tensor_eff_rank1,
                            ref_embedding=embedding_query,
                        )

                query_column, rank1_column = st.columns(
                    2
                )

                with query_column:
                    st.markdown(
                        "**🔵 Logo Query — CNN**"
                    )
                    query_panels = st.columns(3)
                    query_panels[0].image(
                        image_query,
                        caption="Asli",
                        use_container_width=True,
                    )
                    query_panels[1].image(
                        heatmap_to_rgb(
                            gray_query
                        ),
                        caption="Heatmap",
                        use_container_width=True,
                    )
                    query_panels[2].image(
                        overlay_query,
                        caption="Overlay",
                        use_container_width=True,
                    )

                with rank1_column:
                    st.markdown(
                        f"**🔴 Rank-1 — CNN: {rank1_name}**"
                    )

                    if (
                        rank1_image_cam is not None
                        and overlay_rank1 is not None
                        and gray_rank1 is not None
                    ):
                        rank1_panels = st.columns(3)

                        rank1_panels[0].image(
                            rank1_image_cam,
                            caption="Asli",
                            use_container_width=True,
                        )
                        rank1_panels[1].image(
                            heatmap_to_rgb(
                                gray_rank1
                            ),
                            caption="Heatmap",
                            use_container_width=True,
                        )
                        rank1_panels[2].image(
                            overlay_rank1,
                            caption="Overlay",
                            use_container_width=True,
                        )
                    else:
                        st.warning(
                            "Gambar kandidat Rank-1 tidak "
                            "dapat diresolusi secara unik."
                        )

            except Exception as exc:
                st.warning(
                    "EigenGradCAM tidak dapat dibuat "
                    "untuk pengujian ini."
                )
                st.caption(str(exc))

        st.divider()

    st.markdown(
        f'<div class="djki-section-title">TOP-{len(top_indices)} KANDIDAT DARI DATABASE PDKI</div>',
        unsafe_allow_html=True,
    )

    result_rows: list[dict[str, Any]] = []

    # Kumpulkan semua data dulu
    for rank, (index, final_score, embedding_score, hsv_score, is_above) in enumerate(
        zip(top_indices, top_final_scores, top_embedding_scores, top_hsv_scores, above_threshold),
        start=1,
    ):
        metadata = filtered_metadata[int(index)]
        image_path = resolve_image_path(metadata, unique_filename_index)
        brand_name = safe_text(metadata.get("nama_merek"), "N/A")[:120]
        owner = safe_text(metadata.get("owner_name"))[:120]
        application_number = safe_text(metadata.get("nomor_permohonan"))
        nice_class = safe_text(metadata.get("kelas_nice"))
        application_date = safe_text(metadata.get("tanggal_permohonan"))
        status = safe_text(metadata.get("status_permohonan"), "N/A")
    
        result_rows.append({
            "rank": rank,
            "score_final": float(final_score),
            "score_embedding": float(embedding_score),
            "score_hsv": float(hsv_score),
            "above_threshold": bool(is_above),
            "nama_merek": brand_name,
            "owner": owner,
            "nomor_permohonan": application_number,
            "kelas_nice": nice_class,
            "tanggal_permohonan": application_date,
            "status": status,
            "image_path": image_path,
        })
    
    # Render grid 3 kolom
    COLS_PER_ROW = 5
    for row_start in range(0, len(result_rows), COLS_PER_ROW):
        cols = st.columns(COLS_PER_ROW)
        for col_idx, result in enumerate(
            result_rows[row_start:row_start + COLS_PER_ROW]
        ):
            with cols[col_idx]:
                status_lower = result["status"].lower()
                if "didaftar" in status_lower:
                    badge_bg = "#16A34A"
                    badge_text = "Didaftar"
                elif "tolak" in status_lower:
                    badge_bg = COLOR_RED_WARN
                    badge_text = "Ditolak"
                elif any(t in status_lower for t in ("pengumuman", "substantif", "proses")):
                    badge_bg = COLOR_ORANGE
                    badge_text = "Proses"
                else:
                    badge_bg = COLOR_MUTED
                    badge_text = result['status'][:20]
     
                threshold_color = COLOR_RED_WARN if result["above_threshold"] else "#16A34A"
                threshold_text = "Di atas threshold" if result["above_threshold"] else "Di bawah threshold"
     
                pct_final = result['score_final'] * 100
                pct_emb   = result['score_embedding'] * 100
                pct_hsv   = result['score_hsv'] * 100
     
                # Gambar dengan tinggi seragam
                if result["image_path"] is not None:
                    try:
                        img = Image.open(result["image_path"]).convert("RGB")
                        # Buat canvas putih 1:1 agar semua gambar seragam
                        canvas = Image.new("RGB", (300, 300), (255, 255, 255))
                        img.thumbnail((300, 300), Image.LANCZOS)
                        offset = (
                            (300 - img.width) // 2,
                            (300 - img.height) // 2,
                        )
                        canvas.paste(img, offset)
                        st.image(canvas, use_container_width=True)
                    except Exception:
                        st.markdown(
                            "<div style='height:120px; background:#F1F5F9;"
                            "display:flex; align-items:center; justify-content:center;"
                            "color:#94A3B8; font-size:0.75rem; border-radius:8px;'>"
                            "Tidak tersedia</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<div style='height:120px; background:#F1F5F9;"
                        "display:flex; align-items:center; justify-content:center;"
                        "color:#94A3B8; font-size:0.75rem; border-radius:8px;'>"
                        "Tidak tersedia</div>",
                        unsafe_allow_html=True
                    )
     
                owner_text = result['owner'] if result['owner'] else '-'
                st.markdown(
                    f"<div style='border-top:3px solid {COLOR_NAVY};"
                    f"background:white; border-radius:0 0 10px 10px;"
                    f"padding:8px 10px; box-shadow:0 2px 8px rgba(29,45,92,0.10);'>"
     
                    # Rank + badge status
                    f"<div style='display:flex; justify-content:space-between;"
                    f"align-items:center; margin-bottom:4px;'>"
                    f"<span style='font-weight:800; color:{COLOR_NAVY};"
                    f"font-size:0.82rem;'>Rank {result['rank']}</span>"
                    f"<span style='background:{badge_bg}; color:white;"
                    f"border-radius:4px; padding:2px 7px; font-size:0.65rem;"
                    f"font-weight:600;'>{badge_text}</span></div>"
     
                    # Threshold
                    f"<div style='font-size:0.70rem; color:{threshold_color};"
                    f"margin-bottom:5px; font-weight:600;'>{threshold_text}</div>"
     
                    # Nama merek
                    f"<div style='font-size:0.82rem; font-weight:700;"
                    f"color:{COLOR_TEXT}; margin-bottom:2px;"
                    f"overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>"
                    f"{result['nama_merek']}</div>"
     
                    # Pemilik
                    f"<div style='font-size:0.72rem; color:{COLOR_MUTED};"
                    f"margin-bottom:6px; overflow:hidden; text-overflow:ellipsis;"
                    f"white-space:nowrap;'>{owner_text}</div>"
     
                    # Garis pemisah + metadata
                    f"<div style='border-top:1px solid #E2E8F0; padding-top:5px; margin-bottom:6px;'>"
                    f"<div style='font-size:0.70rem; color:{COLOR_MUTED}; margin-bottom:2px;'>"
                    f"<b>No. Permohonan:</b> {result['nomor_permohonan']}</div>"
                    f"<div style='font-size:0.70rem; color:{COLOR_MUTED}; margin-bottom:2px;'>"
                    f"<b>Kelas NICE:</b> {result['kelas_nice']}</div>"
                    f"<div style='font-size:0.70rem; color:{COLOR_MUTED}; margin-bottom:0;'>"
                    f"<b>Tgl. Permohonan:</b> {result['tanggal_permohonan']}</div></div>"
     
                    # Kotak skor
                    f"<div style='background:{COLOR_NAVY}; border-radius:8px; padding:8px 10px;'>"
     
                    # Baris 1: label + persentase akhir
                    f"<div style='display:flex; justify-content:space-between;"
                    f"align-items:center; margin-bottom:5px;'>"
                    f"<span style='color:{COLOR_YELLOW}; font-size:0.68rem;"
                    f"font-weight:700;'>TINGKAT KEMIRIPAN</span>"
                    f"<span style='color:{COLOR_YELLOW}; font-weight:800;"
                    f"font-size:0.90rem;'>{pct_final:.1f}%</span></div>"
     
                    # Baris 2: Fusion | Bentuk | Warna
                    f"<div style='display:flex; justify-content:space-between;"
                    f"border-top:1px solid rgba(255,255,255,0.15); padding-top:5px;'>"
     
                    f"<div style='text-align:center; flex:1;'>"
                    f"<div style='color:rgba(255,255,255,0.65); font-size:0.60rem;"
                    f"margin-bottom:2px;'>Fusion</div>"
                    f"<div style='color:white; font-weight:700; font-size:0.74rem;'>"
                    f"{result['score_final']:.4f}</div></div>"
     
                    f"<div style='width:1px; background:rgba(255,255,255,0.15);'></div>"
     
                    f"<div style='text-align:center; flex:1;'>"
                    f"<div style='color:rgba(255,255,255,0.65); font-size:0.60rem;"
                    f"margin-bottom:2px;'>Bentuk</div>"
                    f"<div style='color:white; font-weight:600; font-size:0.74rem;'>"
                    f"{result['score_embedding']:.4f}</div></div>"
     
                    f"<div style='width:1px; background:rgba(255,255,255,0.15);'></div>"
     
                    f"<div style='text-align:center; flex:1;'>"
                    f"<div style='color:rgba(255,255,255,0.65); font-size:0.60rem;"
                    f"margin-bottom:2px;'>Warna</div>"
                    f"<div style='color:white; font-weight:600; font-size:0.74rem;'>"
                    f"{result['score_hsv']:.4f}</div></div>"
     
                    f"</div></div></div>",
                    unsafe_allow_html=True
                )
     
        st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="djki-section-title">EKSPOR HASIL</div>', unsafe_allow_html=True)

    report_lines = [
        "HASIL ANALISIS KEMIRIPAN LOGO MEREK DAGANG",
        "=" * 72,
        f"Versi aplikasi     : {APP_VERSION}",
        f"Query               : {uploaded.name}",
        f"Threshold operasional: {threshold:.2f}",
        f"Alpha fusion        : {alpha:.2f}",
        f"Bobot embedding     : {alpha:.0%}",
        f"Bobot HSV           : {1 - alpha:.0%}",
        f"Kandidat diperiksa  : {len(result_rows)}",
        f"Di atas threshold   : {above_count}",
        (
            "Filter Kelas Nice   : "
            + (
                ", ".join(
                    map(
                        str,
                        sorted(
                            requested_classes
                        ),
                    )
                )
                if requested_classes
                else "Semua kelas"
            )
        ),
        "",
        (
            f"{'Rank':<5} "
            f"{'Final':<9} "
            f"{'Embed':<9} "
            f"{'HSV':<9} "
            f"{'Threshold':<12} "
            f"{'No. Permohonan':<22} "
            f"Nama Merek"
        ),
        "-" * 110,
    ]

    for row in result_rows:
        report_lines.append(
            f"{row['rank']:<5} "
            f"{row['score_final']:<9.4f} "
            f"{row['score_embedding']:<9.4f} "
            f"{row['score_hsv']:<9.4f} "
            f"{('YA' if row['above_threshold'] else 'TIDAK'):<12} "
            f"{row['nomor_permohonan']:<22} "
            f"{row['nama_merek']}"
        )

    report_lines.extend(
        [
            "",
            "=" * 72,
            (
                "DISCLAIMER: Output sistem DSS dan "
                "bukan keputusan hukum."
            ),
            (
                "Verifikasi oleh pemeriksa merek "
                "tetap diperlukan."
            ),
            (
                "Nilai alpha dan threshold final "
                "harus didukung evaluasi penelitian."
            ),
        ]
    )

    txt_column, csv_column = st.columns(2)

    with txt_column:
        st.download_button(
            label="Download Laporan (.txt)",
            data="\n".join(
                report_lines
            ),
            file_name=(
                "analisis_"
                f"{Path(uploaded.name).stem}.txt"
            ),
            mime="text/plain",
            use_container_width=True,
        )

    with csv_column:
        st.download_button(
            label="Download Data (.csv)",
            data=build_csv_report(
                result_rows
            ),
            file_name=(
                "analisis_"
                f"{Path(uploaded.name).stem}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

if __name__ == "__main__":
    main()
