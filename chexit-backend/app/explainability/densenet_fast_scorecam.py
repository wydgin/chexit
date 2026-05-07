from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf


def _normalize_cam_to_unit(cam: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    cam = np.maximum(cam.astype(np.float32), 0.0)
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin < epsilon:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cmin) / (cmax - cmin + epsilon)


def compute_densenet_fast_scorecam(
    model: tf.keras.Model,
    x: np.ndarray,
    *,
    target_class: int,
    target_layer_name: str = "conv5_block16_concat",
    batch_size: int = 16,
    max_channels: int = 256,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Fast Score-CAM for DenseNet-style binary classifiers.
    Returns normalized CAM in [0, 1] and timing metadata.
    """
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "densenet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise ValueError("DenseNet backbone layer not found in model.")

    target_layer = base_model.get_layer(target_layer_name)
    activation_model = tf.keras.Model(inputs=base_model.input, outputs=target_layer.output)

    t_act0 = time.perf_counter()
    activations = activation_model(x, training=False)
    timings["extract_activations"] = time.perf_counter() - t_act0

    input_shape = tf.shape(x)[1:3]
    act_resized = tf.image.resize(activations, input_shape)[0]  # (H, W, C)
    mins = tf.reduce_min(act_resized, axis=[0, 1])
    maxs = tf.reduce_max(act_resized, axis=[0, 1])
    act_normalized = (act_resized - mins) / (maxs - mins + 1e-10)

    variances = tf.math.reduce_variance(act_normalized, axis=[0, 1])
    n_channels = int(act_normalized.shape[-1])
    max_channels = min(int(max_channels), n_channels)
    top_indices = tf.argsort(variances, direction="DESCENDING")[:max_channels]
    timings["channels_total"] = float(n_channels)
    timings["channels_used"] = float(max_channels)

    scores = np.zeros(max_channels, dtype=np.float32)
    t_mask_fwd0 = time.perf_counter()
    for i in range(0, len(top_indices), int(batch_size)):
        end_idx = min(i + int(batch_size), len(top_indices))
        current_indices = top_indices[i:end_idx]

        maps = tf.gather(act_normalized, current_indices, axis=-1)
        maps = tf.transpose(maps, [2, 0, 1])
        maps = tf.expand_dims(maps, axis=-1)
        masked_batch = x * maps
        preds = model(masked_batch, training=False).numpy().reshape(-1)
        if target_class == 1:
            scores[i:end_idx] = preds
        else:
            scores[i:end_idx] = 1.0 - preds
    timings["masked_forwards"] = time.perf_counter() - t_mask_fwd0

    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    cam = np.zeros(input_shape.numpy(), dtype=np.float32)
    for idx, weight in enumerate(scores):
        channel_idx = top_indices[idx]
        cam += weight * act_normalized[..., channel_idx].numpy()
    norm_cam = _normalize_cam_to_unit(cam)
    timings["total"] = time.perf_counter() - t0
    return norm_cam, timings

