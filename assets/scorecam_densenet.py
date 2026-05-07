"""
DenseNet Score-CAM for single-image TB diagnosis workflow.

Web-app friendly:
- importable predictor object that can stay in memory across requests
- single-image CLI entrypoint with input/output paths
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import cm
import sys

BACKEND_DIR = Path(__file__).resolve().parent.parent / "chexit-backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.explainability.densenet_fast_scorecam import compute_densenet_fast_scorecam

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "densenet_tb_output"
WEIGHTS_DIR = OUTPUT_DIR / "weights"
IMG_SIZE = 256

PathLike = Union[str, Path]
_PREDICTOR_CACHE: Dict[str, "DenseNetScoreCamPredictor"] = {}


def build_densenet_model() -> tf.keras.Model:
    base = tf.keras.applications.DenseNet121(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None,
        pooling=None,
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.4, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(
        512,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="head_dense_1",
    )(x)
    x = tf.keras.layers.Dropout(0.4, name="head_dropout_final")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="tb_output")(x)
    return tf.keras.Model(inputs, out, name="densenet121_tb_classifier")


@dataclass
class DenseNetScoreCamResult:
    output_path: str
    predicted_probability: float
    predicted_label: int


class DenseNetScoreCamPredictor:
    def __init__(
        self,
        model: tf.keras.Model,
        target_layer_name: str = "conv5_block16_concat",
        batch_size: int = 16,
        max_channels: int = 256,
    ):
        self.model = model
        self.target_layer_name = target_layer_name
        self.batch_size = batch_size
        self.max_channels = max_channels

    @classmethod
    def from_weights(
        cls,
        weights_path: PathLike,
        target_layer_name: str = "conv5_block16_concat",
        batch_size: int = 16,
        max_channels: int = 256,
    ) -> "DenseNetScoreCamPredictor":
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        model = build_densenet_model()
        model.load_weights(weights_path)
        return cls(
            model=model,
            target_layer_name=target_layer_name,
            batch_size=batch_size,
            max_channels=max_channels,
        )

    def diagnose_and_save(
        self,
        input_image_path: PathLike,
        output_image_path: PathLike,
        overlay_image_path: Optional[PathLike] = None,
    ) -> DenseNetScoreCamResult:
        input_image_path = Path(input_image_path)
        output_image_path = Path(output_image_path)
        seg_bgr = cv2.imread(str(input_image_path))
        if seg_bgr is None:
            raise FileNotFoundError(f"Unable to read input image: {input_image_path}")
        seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        seg_resized = cv2.resize(seg_rgb, (IMG_SIZE, IMG_SIZE))

        if overlay_image_path:
            overlay_bgr = cv2.imread(str(overlay_image_path))
            if overlay_bgr is not None:
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
                overlay_resized = cv2.resize(overlay_rgb, (IMG_SIZE, IMG_SIZE))
            else:
                overlay_resized = seg_resized
        else:
            overlay_resized = seg_resized

        x_img = np.expand_dims(seg_resized, axis=0).astype(np.float32)
        x_img = tf.keras.applications.densenet.preprocess_input(x_img)
        pred_prob = float(self.model(x_img, training=False).numpy()[0][0])
        pred_label = int(pred_prob >= 0.5)

        heatmap, _ = compute_densenet_fast_scorecam(
            self.model,
            x_img,
            target_class=pred_label,
            target_layer_name=self.target_layer_name,
            batch_size=self.batch_size,
            max_channels=self.max_channels,
        )
        heat_rgb = (cm.jet(heatmap)[..., :3] * 255).astype(np.float32)
        lung_mask = (np.max(seg_resized, axis=-1, keepdims=True) > 10).astype(np.float32)
        heatmap_expanded = np.expand_dims(heatmap, axis=-1)
        alpha_map = np.clip((heatmap_expanded - 0.25) / 0.75, 0.0, 1.0) * 0.65
        final_alpha = alpha_map * lung_mask
        base_float = overlay_resized.astype(np.float32)
        blended = (heat_rgb * final_alpha + base_float * (1.0 - final_alpha)).astype(np.uint8)

        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return DenseNetScoreCamResult(
            output_path=str(output_image_path),
            predicted_probability=pred_prob,
            predicted_label=pred_label,
        )


def get_cached_predictor(
    weights_path: PathLike,
    target_layer_name: str = "conv5_block16_concat",
    batch_size: int = 16,
    max_channels: int = 256,
) -> DenseNetScoreCamPredictor:
    key = f"{Path(weights_path).resolve()}::{target_layer_name}::{batch_size}::{max_channels}"
    predictor = _PREDICTOR_CACHE.get(key)
    if predictor is None:
        predictor = DenseNetScoreCamPredictor.from_weights(
            weights_path=weights_path,
            target_layer_name=target_layer_name,
            batch_size=batch_size,
            max_channels=max_channels,
        )
        _PREDICTOR_CACHE[key] = predictor
    return predictor


def generate_scorecam_heatmap(
    input_image_path: PathLike,
    output_image_path: PathLike,
    predictor: Optional[DenseNetScoreCamPredictor] = None,
    weights_path: Optional[PathLike] = None,
    overlay_image_path: Optional[PathLike] = None,
    use_cache: bool = True,
    target_layer_name: str = "conv5_block16_concat",
    batch_size: int = 16,
    max_channels: int = 256,
) -> DenseNetScoreCamResult:
    if predictor is None:
        resolved_weights = (
            Path(weights_path) if weights_path else (WEIGHTS_DIR / "fold_1_phase2_best.weights.h5")
        )
        if use_cache:
            predictor = get_cached_predictor(
                weights_path=resolved_weights,
                target_layer_name=target_layer_name,
                batch_size=batch_size,
                max_channels=max_channels,
            )
        else:
            predictor = DenseNetScoreCamPredictor.from_weights(
                weights_path=resolved_weights,
                target_layer_name=target_layer_name,
                batch_size=batch_size,
                max_channels=max_channels,
            )

    return predictor.diagnose_and_save(
        input_image_path=input_image_path,
        output_image_path=output_image_path,
        overlay_image_path=overlay_image_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DenseNet Score-CAM heatmap for one image."
    )
    parser.add_argument("--input", required=True, help="Input segmented image path.")
    parser.add_argument("--output", required=True, help="Output heatmap image path.")
    parser.add_argument(
        "--weights",
        default=str(WEIGHTS_DIR / "fold_1_phase2_best.weights.h5"),
        help="Model weights path.",
    )
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional original image path used as the visualization base.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for Score-CAM masking."
    )
    parser.add_argument(
        "--max-channels",
        type=int,
        default=256,
        help="Top activation channels used for fast Score-CAM.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable predictor cache (loads fresh model each run).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = generate_scorecam_heatmap(
        input_image_path=args.input,
        output_image_path=args.output,
        weights_path=args.weights,
        overlay_image_path=args.overlay,
        use_cache=not args.no_cache,
        batch_size=args.batch_size,
        max_channels=args.max_channels,
    )
    print(
        f"Saved heatmap: {result.output_path} | "
        f"pred={result.predicted_label} prob={result.predicted_probability:.4f}"
    )


if __name__ == "__main__":
    main()

