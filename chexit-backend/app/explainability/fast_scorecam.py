from __future__ import annotations

import time
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf


def _normalize_minmax_hw(maps_hw: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    out = np.empty_like(maps_hw, dtype=np.float32)
    for i in range(maps_hw.shape[0]):
        a = maps_hw[i].astype(np.float32)
        amin, amax = float(a.min()), float(a.max())
        if amax - amin < epsilon:
            out[i] = 0.0
        else:
            out[i] = (a - amin) / (amax - amin + epsilon)
    return out


def _normalize_cam_to_unit(cam: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    cam = np.maximum(cam.astype(np.float32), 0.0)
    cmin, cmax = float(cam.min()), float(cam.max())
    if cmax - cmin < epsilon:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cmin) / (cmax - cmin + epsilon)


def _predict_probs(model: tf.keras.Model, x: np.ndarray) -> Tuple[np.ndarray, int]:
    y = model.predict(x, verbose=0)
    if y.ndim == 2 and y.shape[-1] == 1:
        return y, 1
    if y.ndim == 2 and y.shape[-1] > 1:
        return y, int(y.shape[-1])
    if y.ndim == 1:
        return y.reshape(-1, 1), 1
    raise ValueError(f"Unexpected model output shape: {y.shape}")


def _gather_target_score(y: np.ndarray, target_class: int, num_classes: int) -> np.ndarray:
    if num_classes == 1:
        p_pos = y.reshape(-1).astype(np.float32)
        p_neg = 1.0 - p_pos
        return p_pos if target_class == 1 else p_neg
    return y[:, target_class].astype(np.float32)


def _resolve_target_layer(
    model: tf.keras.Model,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]],
) -> tf.keras.layers.Layer:
    if penultimate_layer is not None:
        if isinstance(penultimate_layer, str):
            # Try top-level first
            try:
                return model.get_layer(penultimate_layer)
            except Exception:
                # Then nested models
                for layer in model.layers:
                    if isinstance(layer, tf.keras.Model):
                        try:
                            return layer.get_layer(penultimate_layer)
                        except Exception:
                            continue
                raise
        return penultimate_layer

    gap_idx: Optional[int] = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_idx = i
            break
    if gap_idx is None or gap_idx < 1:
        raise ValueError("No GlobalAveragePooling2D found.")
    return model.layers[gap_idx - 1]


def compute_fast_scorecam(
    model: tf.keras.Model,
    seed_input: np.ndarray,
    *,
    target_class: int,
    penultimate_layer: Optional[Union[str, tf.keras.layers.Layer]] = None,
    batch_size: int = 32,
    max_channels: Optional[int] = 256,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    t0 = time.perf_counter()
    timings: Dict[str, float] = {}

    feat_layer = _resolve_target_layer(model, penultimate_layer)
    t_act0 = time.perf_counter()
    try:
        if isinstance(feat_layer, tf.keras.Model):
            acts = feat_layer(tf.convert_to_tensor(seed_input), training=False).numpy()
        else:
            feat_model = tf.keras.Model(model.inputs, feat_layer.output, name="scorecam_features")
            acts = feat_model.predict(seed_input, verbose=0)
    except Exception:
        # Keras 3 can report disconnected outputs for nested backbone layers.
        # Fallback: locate the nested parent model that owns the layer and run
        # activation extraction directly in that subgraph.
        layer_name = getattr(feat_layer, "name", None)
        nested_parent = None
        nested_target = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model):
                try:
                    nested_target = layer.get_layer(layer_name) if layer_name else None
                    nested_parent = layer
                    break
                except Exception:
                    continue
        if nested_parent is None or nested_target is None:
            raise
        nested_act_model = tf.keras.Model(
            inputs=nested_parent.input,
            outputs=nested_target.output,
            name="scorecam_nested_features",
        )
        acts = nested_act_model(tf.convert_to_tensor(seed_input), training=False).numpy()
    timings["extract_activations"] = time.perf_counter() - t_act0

    if acts.ndim != 4:
        raise ValueError(f"Expected 4D activations, got {acts.shape}")

    _, _, _, n_ch = acts.shape
    _, H, W, _ = seed_input.shape
    acts_ch = np.transpose(acts[0], (2, 0, 1))

    t_up0 = time.perf_counter()
    ups = np.empty((n_ch, H, W), dtype=np.float32)
    for c in range(n_ch):
        ups[c] = cv2.resize(acts_ch[c], (W, H), interpolation=cv2.INTER_LINEAR)
    timings["upsample_maps"] = time.perf_counter() - t_up0

    masks = _normalize_minmax_hw(ups)
    selected_masks = masks
    if max_channels is not None and int(max_channels) > 0 and int(max_channels) < n_ch:
        variances = np.var(masks, axis=(1, 2))
        top_idx = np.argsort(-variances)[: int(max_channels)]
        selected_masks = masks[top_idx]

    timings["channels_total"] = float(n_ch)
    timings["channels_used"] = float(selected_masks.shape[0])

    x0 = seed_input.astype(np.float32)
    weights = []
    n_sel = int(selected_masks.shape[0])

    t_mask_fwd0 = time.perf_counter()
    for start in range(0, n_sel, int(batch_size)):
        end = min(start + int(batch_size), n_sel)
        bsz = end - start
        batch = np.empty((bsz, H, W, 3), dtype=np.float32)
        for j in range(bsz):
            batch[j] = x0[0] * selected_masks[start + j][..., np.newaxis]
        yb, n_out = _predict_probs(model, batch)
        scores = _gather_target_score(yb, target_class, n_out)
        weights.extend([float(s) for s in scores])
    timings["masked_forwards"] = time.perf_counter() - t_mask_fwd0

    w_vec = np.asarray(weights, dtype=np.float32).reshape(n_sel)
    cam = np.tensordot(w_vec, selected_masks, axes=([0], [0]))
    cam = np.maximum(cam.astype(np.float32), 0.0)
    norm_cam = _normalize_cam_to_unit(cam)
    timings["total"] = time.perf_counter() - t0
    return cam, norm_cam, timings

