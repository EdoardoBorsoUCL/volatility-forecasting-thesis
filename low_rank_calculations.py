
from __future__ import annotations
import numpy as np
import tensorflow as tf
from gpflow.config import default_float, default_jitter

# ---------------------------------------------------------------------
# dtype / small helpers
# ---------------------------------------------------------------------
def _tf_float() -> tf.DType:
    return default_float()

def _to_float(x) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.cast(x, _tf_float()) if x.dtype != _tf_float() else x


# ---------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------
def _draw_indices(n: int | tf.Tensor, l: int | tf.Tensor, need_inv: bool = False):
    """
    Draw l indices from 0..n-1 without replacement.
    Returns (idx_sampled, idx_not_sampled[, inv_map]), where inv_map is the inverse permutation:
      inv_map[v] = position of value v in the shuffled 'idx'.
    """
    n = tf.convert_to_tensor(n, dtype=tf.int32)
    l = tf.convert_to_tensor(l, dtype=tf.int32)

    idx = tf.random.shuffle(tf.range(n, dtype=tf.int32))  # permutation of 0..n-1
    idx_sampled, idx_not_sampled = tf.split(idx, [l, n - l])

    if need_inv:
        # inverse permutation: arg-sort of idx (ascending)
        inv_map = tf.argsort(idx, stable=True)
        return idx_sampled, idx_not_sampled, inv_map
    else:
        return idx_sampled, idx_not_sampled


def _draw_n_rademacher_samples(n: int | tf.Tensor, seed: tf.Tensor | None = None):
    """
    Draw n Rademacher variables in {+1, -1}.
    """
    n = tf.convert_to_tensor(n, dtype=tf.int32)
    if seed is None:
        u = tf.random.uniform([n], dtype=_tf_float())
    else:
        u = tf.random.stateless_uniform([n], seed=tf.cast(seed, tf.int32), dtype=_tf_float())
    return tf.where(u <= 0.5, tf.ones([n], dtype=_tf_float()), -tf.ones([n], dtype=_tf_float()))


def _draw_n_gaussian_samples(n: int | tf.Tensor, seed: tf.Tensor | None = None):
    """
    Draw n standard normal samples.
    """
    n = tf.convert_to_tensor(n, dtype=tf.int32)
    if seed is None:
        return tf.random.normal([n], dtype=_tf_float())
    else:
        return tf.random.stateless_normal([n], seed=tf.cast(seed, tf.int32), dtype=_tf_float())


def _draw_n_sparse_gaussian_samples(n: int | tf.Tensor, s: float | tf.Tensor, seed: tf.Tensor | None = None):
    """
    Draw n sparse Gaussians with P(N(0,1)) = 1/s, else 0.
    """
    n = tf.convert_to_tensor(n, dtype=tf.int32)
    s = _to_float(s)
    if seed is None:
        u = tf.random.uniform([n], dtype=_tf_float())
        z = tf.random.normal([n], dtype=_tf_float())
    else:
        seed = tf.cast(seed, tf.int32)
        u = tf.random.stateless_uniform([n], seed=seed, dtype=_tf_float())
        z = tf.random.stateless_normal([n],  seed=seed, dtype=_tf_float())
    return tf.where(u <= 1.0 / s, z, tf.zeros([n], dtype=_tf_float()))


# ---------------------------------------------------------------------
# Nystrom features
# ---------------------------------------------------------------------
def Nystrom_map(X, kern, nys_samples=None, num_components=None):
    """
    Compute Nyström features X_nys ≈ K(X, Z) U diag(S)^(-1/2), robust in float32.
    X: (N, D); Z = nys_samples if provided else uniform subset of X.
    Returns: (N, m) with m = #Nyström points.
    """
    f = default_float()
    X = tf.cast(X, f)
    N = tf.shape(X)[0]

    if nys_samples is None and num_components is None:
        raise ValueError("One of num_components or nys_samples must be provided.")

    if nys_samples is None:
        # uniform subset
        idx = tf.random.shuffle(tf.range(N))[: int(num_components)]
        Z = tf.gather(X, idx, axis=0)
    else:
        Z = tf.cast(nys_samples, f)

    m = tf.shape(Z)[0]

    # Kernel blocks
    W = kern(Z, Z)            # (m, m)
    Kxz = kern(X, Z)          # (N, m)

    # Symmetrize + scale-aware jitter
    W = 0.5 * (W + tf.transpose(W))
    mean_diag = tf.reduce_mean(tf.linalg.diag_part(W))
    eps = tf.maximum(mean_diag * tf.cast(1e-6, f), tf.cast(1e-6, f))
    W = W + (default_jitter() + eps) * tf.eye(m, dtype=f)

    # SVD (more robust than eigh on near-PSD)
    # tf.linalg.svd returns s, u, v if compute_uv=True (default)
    s, u, v = tf.linalg.svd(W, full_matrices=False, compute_uv=True)
    # Clamp small/negative to avoid nan in sqrt
    s = tf.maximum(s, tf.cast(1e-8, f))
    d_inv_sqrt = tf.math.rsqrt(s)                   # 1/sqrt(s)

    X_nys = tf.matmul(Kxz, u) * d_inv_sqrt[None, :] # (N, m)
    return X_nys



# ---------------------------------------------------------------------
# Low-rank Hadamard products
# ---------------------------------------------------------------------
def lr_hadamard_prod(A: tf.Tensor, B: tf.Tensor) -> tf.Tensor:
    """
    Low-rank equivalent of Hadamard product (outer product of features).
    A: (..., k1), B: (..., k2) -> (..., k1*k2)
    """
    A = _to_float(A)
    B = _to_float(B)
    C = tf.matmul(tf.expand_dims(A, axis=-1), tf.expand_dims(B, axis=-2))  # (..., k1, k2)
    flat_last = tf.reduce_prod(tf.shape(C)[-2:])
    return tf.reshape(C, tf.concat([tf.shape(C)[:-2], [flat_last]], axis=0))


def lr_hadamard_prod_subsample(A: tf.Tensor, B: tf.Tensor, num_components: int, seed: tf.Tensor | None = None) -> tf.Tensor:
    """
    Subsampled low-rank Hadamard product.
    Returns (..., num_components)
    """
    A = _to_float(A)
    B = _to_float(B)
    batch_shape = tf.shape(A)[:-1]
    k1 = tf.shape(A)[-1]
    k2 = tf.shape(B)[-1]

    idx1 = tf.reshape(tf.range(k1, dtype=tf.int32), [1, -1, 1])
    idx2 = tf.reshape(tf.range(k2, dtype=tf.int32), [-1, 1, 1])
    combinations = tf.concat([idx1 + tf.zeros_like(idx2), tf.zeros_like(idx1) + idx2], axis=2)
    combinations = tf.reshape(combinations, [-1, 2])
    combinations = tf.random.shuffle(combinations)

    select = combinations[:num_components]
    A_sel = tf.gather(A, select[:, 0], axis=-1)
    B_sel = tf.gather(B, select[:, 1], axis=-1)
    C = tf.reshape(A_sel * B_sel, [-1, num_components])

    D = tf.expand_dims(_draw_n_rademacher_samples(num_components, seed=seed), axis=0)
    C = C * D
    return tf.reshape(C, tf.concat([batch_shape, [num_components]], axis=0))


def lr_hadamard_prod_sparse(A: tf.Tensor, B: tf.Tensor, num_components: int, sparse_scale: str, seed: tf.Tensor | None = None) -> tf.Tensor:
    """
    Very Sparse Johnson–Lindenstrauss transform variant for Hadamard product.
    A: (..., k1), B: (..., k2) -> (..., num_components)
    """
    A = _to_float(A)
    B = _to_float(B)

    batch_shape = tf.shape(A)[:-1]
    k1 = tf.shape(A)[-1]
    k2 = tf.shape(B)[-1]

    idx1 = tf.reshape(tf.range(k1, dtype=tf.int32), [1, -1, 1])
    idx2 = tf.reshape(tf.range(k2, dtype=tf.int32), [-1, 1, 1])
    combinations = tf.reshape(tf.concat([idx1 + tf.zeros_like(idx2), tf.zeros_like(idx1) + idx2], axis=2), [-1, 2])

    D = k1 * k2
    rand_matrix_size = D * num_components

    if sparse_scale == "log":
        s = _to_float(D) / tf.math.log(_to_float(D))
    elif sparse_scale == "sqrt":
        s = tf.sqrt(_to_float(D))
    else:
        # keep "lin" case in the dispatcher above; default here to sqrt if unknown
        s = tf.sqrt(_to_float(D))

    R = tf.reshape(_draw_n_sparse_gaussian_samples(rand_matrix_size, s, seed=seed), [D, num_components])

    # keep only rows with non-zero entries in R
    idx_result = tf.math.count_nonzero(R, axis=1, dtype=tf.int32) > 0
    idx_combined = tf.boolean_mask(combinations, idx_result, axis=0)
    n_nonzero = tf.shape(idx_combined)[0]

    A_nz = tf.reshape(tf.gather(A, idx_combined[:, 0], axis=-1), [-1, n_nonzero])
    B_nz = tf.reshape(tf.gather(B, idx_combined[:, 1], axis=-1), [-1, n_nonzero])
    C = A_nz * B_nz

    R_nonzero = tf.boolean_mask(R, idx_result, axis=0)   # (n_nonzero, num_components)
    C = tf.matmul(C, R_nonzero)                          # (..., num_components)

    scale = tf.sqrt(s / _to_float(num_components))
    return scale * tf.reshape(C, tf.concat([batch_shape, [num_components]], axis=0))


def lr_hadamard_prod_rand(A: tf.Tensor, B: tf.Tensor, rank_bound: int, sparsity: str = "sqrt", seeds: tf.Tensor | None = None) -> tf.Tensor:
    """
    Wrapper selecting the randomized Hadamard strategy.
    Returns (..., rank_bound)
    """
    if sparsity == "lin":
        return lr_hadamard_prod_subsample(A, B, rank_bound, seed=seeds)
    else:
        return lr_hadamard_prod_sparse(A, B, rank_bound, sparse_scale=sparsity, seed=seeds)
