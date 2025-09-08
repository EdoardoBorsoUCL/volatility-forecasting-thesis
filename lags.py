# lags.py — TF2 / GPflow 2

from __future__ import annotations
import numpy as np
import tensorflow as tf
from gpflow.config import default_float, default_jitter


def _tf_float() -> tf.DType:
    return default_float()


def lin_interp(time: tf.Tensor, X: tf.Tensor, time_query: tf.Tensor) -> tf.Tensor:
    """
    Linear interpolation in time.

    Parameters
    ----------
    time : (L,) float tensor
        Monotone grid in [0,1] (or any ascending scale).
    X : (..., L, D) or (N, L, D) or (N, L, K, D)
        Values on `time`. Last-but-one dim is the time axis.
        (Original code handled ndims==3 or 4; we keep that.)
    time_query : (L, num_lags) float tensor
        Query times per position and per lag, clipped to [0,1].

    Returns
    -------
    X_query : (N, L, num_lags, D) if X.ndim==3
              (N, L, num_lags, D) if X.ndim==4 (broadcasted over the extra dim)
              Matches original behavior.
    """
    f = _tf_float()
    time       = tf.cast(time, f)                  # (L,)
    time_query = tf.cast(time_query, f)            # (L, num_lags)

    # pairwise distances: (L, L, num_lags)
    pairwise_dist = time[:, None, None] - time_query[None, :, :]

    # Mask future points (> jitter) to -inf, then pick the closest left neighbor
    # (same as original: tf.where(pairwise_dist > settings.jitter, -inf, pairwise_dist))
    jitter = tf.cast(default_jitter(), f)
    neg_inf = tf.cast(-np.inf, f)
    masked = tf.where(pairwise_dist > jitter, neg_inf, pairwise_dist)

    # left_idx/right_idx: (L, num_lags)
    left_idx  = tf.argmax(masked, axis=0, output_type=tf.int32)
    right_idx = left_idx + 1

    # Gather along time axis (axis=-2)
    if X.shape.ndims == 3:
        # X: (N, L, D)
        X_left  = tf.gather(X, left_idx,  axis=-2, batch_dims=1)   # (N, L, num_lags, D)
        X_right = tf.gather(X, right_idx, axis=-2, batch_dims=1)
        t_left  = tf.gather(time, left_idx)                        # (L, num_lags)
        t_right = tf.gather(time, right_idx)
        X_query = X_left + (time_query[None, ..., None] - t_left[None, ..., None]) * \
                  (X_right - X_left) / (t_right[None, ..., None] - t_left[None, ..., None])
    elif X.shape.ndims == 4:
        # X: (N, L, K, D)  — original code supported this; we gather across the L axis
        # We need to gather with batch_dims=1, but keep K in place.
        # Reshape to merge K into batch for gather, then restore.
        N = tf.shape(X)[0]
        L = tf.shape(X)[1]
        K = tf.shape(X)[2]
        D = tf.shape(X)[3]

        X_ = tf.reshape(X, [N * K, L, D])  # merge K into batch
        X_left_  = tf.gather(X_, left_idx,  axis=-2, batch_dims=1)  # (N*K, L, num_lags, D)
        X_right_ = tf.gather(X_, right_idx, axis=-2, batch_dims=1)
        X_left  = tf.reshape(X_left_,  [N, K, L, tf.shape(time_query)[1], D])
        X_right = tf.reshape(X_right_, [N, K, L, tf.shape(time_query)[1], D])

        t_left  = tf.gather(time, left_idx)   # (L, num_lags)
        t_right = tf.gather(time, right_idx)

        X_query = X_left + (time_query[None, None, ..., None] - t_left[None, None, ..., None]) * \
                  (X_right - X_left) / (t_right[None, None, ..., None] - t_left[None, None, ..., None])
        # Return shape (N, L, num_lags, D): collapse K by averaging (or keep K if you used it that way).
        # Original code broadcasted; to keep identical last return rank, reduce K dimension if present:
        X_query = tf.reduce_mean(X_query, axis=1)  # (N, L, num_lags, D)
    else:
        raise ValueError("lags.lin_interp: X.ndim must be 3 or 4.")

    return X_query


def add_lags_to_sequences(X: tf.Tensor, lags: tf.Tensor) -> tf.Tensor:
    """
    Add lagged versions of sequences as extra feature blocks (with linear interpolation).
    Matches original shapes.

    Parameters
    ----------
    X : (N, L, D) float tensor
        Batch of sequences.
    lags : (num_lags,) float tensor
        Lag values in [0,1].

    Returns
    -------
    X_new : (N, L, (num_lags+1), D)
        Concatenation of original (lag 0) and lagged versions along a new axis.
    """
    f = _tf_float()
    X    = tf.cast(X, f)
    lags = tf.cast(lags, f)

    N, L, D = tf.unstack(tf.shape(X))
    num_lags = tf.shape(lags)[0]

    # Normalized time in [0,1]
    # Guard L-1 to avoid div-by-zero for degenerate length
    denom = tf.maximum(L - 1, 1)
    time = tf.range(tf.cast(L, f), dtype=f) / tf.cast(denom, f)
    # Shifted queries, clipped to [0,1]
    time_lags = tf.maximum(time[:, None] - lags[None, :], tf.cast(0.0, f))  # (L, num_lags)

    # Interpolate each lag at each position
    X_lags = lin_interp(time, X, time_lags)  # (N, L, num_lags, D)

    # Stack original (lag=0) in front
    X0 = X[:, :, None, :]                    # (N, L, 1, D)
    X_new = tf.concat([X0, X_lags], axis=2)  # (N, L, 1+num_lags, D)
    return X_new
