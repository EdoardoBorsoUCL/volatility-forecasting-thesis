import numpy as np
import tensorflow as tf
import gpflow as gf

TF_DTYPE = gf.config.default_float()  # e.g. tf.float64

# --------------------------
# 1) Inducing *tensors*
# --------------------------
def _sample_inducing_tensors_tf2(sequences, num_inducing, num_levels, increments: bool):
    """
    sequences: np.ndarray or tf.Tensor of shape (N, L, d)
    returns:   np.ndarray of shape (num_inducing, ?, ?, d) depending on flags
    """
    if isinstance(sequences, tf.Tensor):
        sequences_np = sequences.numpy()
    else:
        sequences_np = sequences

    N, L, d = sequences_np.shape
    Z = []

    sequences_select = sequences_np[np.random.choice(N, size=num_inducing, replace=True)]

    for m in range(1, num_levels + 1):
        if increments:
            # pick m indices in [0, L-2], sorted
            obs_idx = [np.random.choice(L - 1, size=(1, m, 1), replace=False) for _ in range(num_inducing)]
            obs_idx = np.sort(np.concatenate(obs_idx, axis=0), axis=1)  # (M, m, 1)

            obs1_select = np.take_along_axis(sequences_select, obs_idx, axis=1)          # (M, m, d)
            obs2_select = np.take_along_axis(sequences_select, obs_idx + 1, axis=1)      # (M, m, d)
            increments_select = np.concatenate(
                (obs1_select[:, :, None, :], obs2_select[:, :, None, :]), axis=2
            )  # (M, m, 2, d)
            Z.append(increments_select)
        else:
            # pick m indices in [0, L-1], sorted
            obs_idx = [np.random.choice(L, size=(1, m, 1), replace=False) for _ in range(num_inducing)]
            obs_idx = np.sort(np.concatenate(obs_idx, axis=0), axis=1)  # (M, m, 1)

            obs_select = np.take_along_axis(sequences_select, obs_idx, axis=1)  # (M, m, d)
            Z.append(obs_select)

    Z = np.concatenate(Z, axis=1)  # increments: (M, sum_m m, 2, d)  else: (M, sum_m m, d)
    return Z


def suggest_initial_inducing_tensors_tf2(
    sequences,
    num_levels: int,
    num_inducing: int,
    labels=None,
    increments: bool = False,
    num_lags: int | None = None,
):
    """
    Returns Z shaped like original code:
      - Start from concatenated levels, reshape to (len_inducing, M, ...), then squeeze+transpose to match.
      - Adds Gaussian jitter at the end.
    """
    if isinstance(sequences, tf.Tensor):
        sequences_np = sequences.numpy()
    else:
        sequences_np = sequences
    N, L, d = sequences_np.shape

    Z_parts = []
    len_inducing = int(num_levels * (num_levels + 1) / 2)

    if labels is not None:
        labels = np.asarray(labels)
        bincount = np.bincount(labels)
        for c, n_c in enumerate(bincount):
            m_c = int(np.floor(float(n_c) / N * num_inducing))
            seq_c = sequences_np[labels == c]
            if m_c > 0 and len(seq_c):
                Z_parts.append(_sample_inducing_tensors_tf2(seq_c, m_c, num_levels, increments))
        num_diff = num_inducing - np.sum([z.shape[0] for z in Z_parts]) if Z_parts else num_inducing
    else:
        num_diff = num_inducing

    if num_diff > 0:
        Z_parts.append(_sample_inducing_tensors_tf2(sequences_np, num_diff, num_levels, increments))

    Z = np.concatenate(Z_parts, axis=0)

    # reshape/transpose exactly like the original
    if increments:
        # Z: (M, sum_m m, 2, d)
        Z = np.squeeze(
            Z.reshape([Z.shape[0], len_inducing, -1, Z.shape[-2], Z.shape[-1]])
             .transpose([1, 0, 2, 3, 4])
        )  # -> (len_inducing, M, ?, 2, d) then squeeze
        if num_lags is not None and num_lags > 0:
            # tile on lag dimension
            Z = np.tile(Z[:, :, :, None, :], (1, 1, 1, num_lags + 1, 1))  # (..., 2, d) -> (..., lag, d)
            Z = Z.reshape([Z.shape[0], Z.shape[1], 2, -1])  # fold lag into last axis
    else:
        # Z: (M, sum_m m, d)
        Z = np.squeeze(
            Z.reshape([Z.shape[0], len_inducing, -1, Z.shape[-1]])
             .transpose([1, 0, 2, 3])
        )  # -> (len_inducing, M, ?, d)
        if num_lags is not None and num_lags > 0:
            Z = np.tile(Z[:, :, :, None, :], (1, 1, num_lags + 1, 1))
            Z = Z.reshape([Z.shape[0], Z.shape[1], -1])

    Z = Z + 0.4 * np.random.randn(*Z.shape)
    return Z  # numpy


# --------------------------
# 2) Inducing *sequences*
# --------------------------
def _sample_inducing_sequences_tf2(sequences, num_inducing, len_inducing):
    """
    sequences: (N, L, d)
    Chooses the *last* contiguous window of length len_inducing for each sampled path,
    respecting any NaN tail markers like original code.
    """
    if isinstance(sequences, tf.Tensor):
        sequences_np = sequences.numpy()
    else:
        sequences_np = sequences

    N, L, d = sequences_np.shape
    Z = []

    sequences_select = sequences_np[np.random.choice(N, size=num_inducing, replace=True)]  # (M, L, d)

    # Find first NaN along time; if none, treat as L
    nans_start = np.argmax(np.any(np.isnan(sequences_select), axis=2), axis=1)  # (M,)
    nans_start[nans_start == 0] = L

    # Choose last index so that window fits (len_inducing)
    last_obs_idx = np.concatenate([
        np.random.choice(range(len_inducing - 1, nans_start[i]), size=(1))
        for i in range(num_inducing)
    ], axis=0)  # (M,)

    # Build indices for contiguous window [last-len+1, ..., last]
    obs_idx = np.stack([last_obs_idx - len_inducing + 1 + i for i in range(len_inducing)], axis=1)[..., None]  # (M, len, 1)

    Z = np.take_along_axis(sequences_select, obs_idx, axis=1)  # (M, len_inducing, d)
    return Z


def suggest_initial_inducing_sequences_tf2(sequences, num_inducing, len_inducing, labels=None):
    if isinstance(sequences, tf.Tensor):
        sequences_np = sequences.numpy()
    else:
        sequences_np = sequences
    N = sequences_np.shape[0]

    Z_parts = []
    if labels is not None:
        labels = np.asarray(labels)
        bincount = np.bincount(labels)
        for c, n_c in enumerate(bincount):
            m_c = int(np.floor(float(n_c) / N * num_inducing))
            seq_c = sequences_np[labels == c]
            if m_c > 0 and len(seq_c):
                Z_parts.append(_sample_inducing_sequences_tf2(seq_c, m_c, len_inducing))
        num_diff = num_inducing - np.sum([z.shape[0] for z in Z_parts]) if Z_parts else num_inducing
    else:
        num_diff = num_inducing

    if num_diff > 0:
        Z_parts.append(_sample_inducing_sequences_tf2(sequences_np, num_diff, len_inducing))

    Z = np.concatenate(Z_parts, axis=0)  # (M, len_inducing, d)
    Z = Z + 0.4 * np.random.randn(*Z.shape)
    return Z  # numpy


# --------------------------
# 3) Lengthscale suggestion
# --------------------------
def suggest_initial_lengthscales_tf2(X, num_samples: int | None = None):
    """
    X: np.ndarray or tf.Tensor of shape (N, D) or (N, L, d); will be flattened to (N*L, d) if 3D.
    Returns a numpy array of per-dim lengthscales (>= 1.0), like the original.
    """
    if isinstance(X, tf.Tensor):
        X_tf = tf.identity(X)
    else:
        X_tf = tf.convert_to_tensor(X)

    # Flatten if sequences:
    if tf.rank(X_tf) == 3:
        N, L, d = X_tf.shape
        X_tf = tf.reshape(X_tf, [-1, d])

    # drop rows with NaNs
    mask = tf.logical_not(tf.reduce_any(tf.math.is_nan(X_tf), axis=1))
    X_tf = tf.boolean_mask(X_tf, mask)
    X_tf = tf.cast(X_tf, TF_DTYPE)

    if num_samples is not None:
        n = tf.shape(X_tf)[0]
        num = tf.minimum(num_samples, n)
        idx = tf.random.shuffle(tf.range(n))[:num]
        X_tf = tf.gather(X_tf, idx, axis=0)

    # pairwise squared distances per-dimension (no kernels used, just heuristic)
    # shape: (n, n, d) -> reshape to (-1, d) and mean over pairs
    X_sq = tf.square(X_tf)
    # ||x - y||^2 = x^2 + y^2 - 2xy, done per-dimension
    dists = (
        X_sq[:, None, :] + X_sq[None, :, :] - 2.0 * (X_tf[:, None, :] * X_tf[None, :, :])
    )
    dists = tf.reshape(dists, [-1, tf.shape(X_tf)[1]])
    l_init = tf.sqrt(tf.reduce_mean(dists, axis=0) * tf.cast(tf.shape(X_tf)[1], TF_DTYPE))

    l_np = l_init.numpy()
    return np.maximum(l_np, 1.0)
