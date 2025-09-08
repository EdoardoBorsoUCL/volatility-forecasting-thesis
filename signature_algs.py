from __future__ import annotations
import numpy as np
import tensorflow as tf
from gpflow.config import default_float

from low_rank_calculations import lr_hadamard_prod_rand


# ----------------------------------------------------------------------------- #
# Small helpers
# ----------------------------------------------------------------------------- #
def _tf_float() -> tf.DType:
    return default_float()

def _one_like(shape, ref_dtype=None):
    dtype = _tf_float() if ref_dtype is None else ref_dtype
    return tf.cast(1.0, dtype) * tf.ones(shape, dtype=dtype)


# ----------------------------------------------------------------------------- #
# First-order signature kernel (dense)
# ----------------------------------------------------------------------------- #
def signature_kern_first_order(M: tf.Tensor, num_levels: int, difference: bool = True) -> tf.Tensor:
    """
    Compute first-order signature kernel levels.

    Parameters
    ----------
    M : tf.Tensor
        Either (n1, L1, n2, L2) or (n, L, L) block Gram tensor
    num_levels : int
        Number of signature levels to compute (returns levels 0..num_levels)
    difference : bool
        If True, differences along the time axis

    Returns
    -------
    K_lvls : tf.Tensor
        Shape (num_levels+1, n1, n2) or (num_levels+1, n)
    """
    if M.shape.ndims == 4:
        # (n1, L1, n2, L2)
        if difference:
            M = (M[:, 1:, :, 1:] + M[:, :-1, :, :-1]
                 - M[:, :-1, :, 1:] - M[:, 1:, :, :-1])
        zeros = tf.reduce_sum(M * 0.0, axis=(1, -1))  # -> (n1, n2)
        K = [zeros + tf.cast(1.0, _tf_float())]
    else:
        # (n, L, L)
        if difference:
            M = (M[:, 1:, 1:] + M[:, :-1, :-1]
                 - M[:, :-1, 1:] - M[:, 1:, :-1])
        zeros = tf.reduce_sum(M * 0.0, axis=(1, -1))  # -> (n,)
        K = [zeros + tf.cast(1.0, _tf_float())]

    # level-1
    K.append(tf.reduce_sum(M, axis=(1, -1)))

    # higher levels via cumsums
    R = M
    for _ in range(2, num_levels + 1):
        R = M * tf.cumsum(tf.cumsum(R, exclusive=True, axis=1), exclusive=True, axis=-1)
        K.append(tf.reduce_sum(R, axis=(1, -1)))

    return tf.stack(K, axis=0)


# ----------------------------------------------------------------------------- #
# Higher-order signature kernel (dense, truncated order)
# ----------------------------------------------------------------------------- #
def signature_kern_higher_order(
    M: tf.Tensor, num_levels: int, order: int = 2, difference: bool = True
) -> tf.Tensor:
    """
    Higher-order (truncated) signature kernel levels.

    Parameters
    ----------
    M : tf.Tensor
        Either (n1, L1, n2, L2) or (n, L, L) block Gram tensor
    num_levels : int
        Number of signature levels to compute (0..num_levels)
    order : int
        Truncation order
    difference : bool
        If True, differences along time axis

    Returns
    -------
    K_lvls : tf.Tensor
        Shape (num_levels+1, n1, n2) or (num_levels+1, n)
    """
    if M.shape.ndims == 4:
        zeros = tf.reduce_sum(M * 0.0, axis=(1, 3))  # -> (n1, n2)
        K = [zeros + tf.cast(1.0, _tf_float())]
    else:
        zeros = tf.reduce_sum(M * 0.0, axis=(1, 2))  # -> (n,)
        K = [zeros + tf.cast(1.0, _tf_float())]

    if difference:
        M = (M[:, 1:, ..., 1:] + M[:, :-1, ..., :-1]
             - M[:, :-1, ..., 1:] - M[:, 1:, ..., :-1])

    # level-1
    K.append(tf.reduce_sum(M, axis=(1, -1)))

    # initialize recursion container with numpy array of tensors (like original)
    R = np.asarray([[M]])
    for i in range(2, num_levels + 1):
        d = min(i, order)
        R_next = np.empty((d, d), dtype=object)

        # (0,0)
        R_next[0, 0] = M * tf.cumsum(
            tf.cumsum(tf.add_n(R.flatten().tolist()), exclusive=True, axis=1),
            exclusive=True, axis=-1
        )

        # first row/col
        for j in range(2, d + 1):
            jf = tf.cast(j, _tf_float())
            R_next[0, j - 1] = (M / jf) * tf.cumsum(tf.add_n(R[:, j - 2].tolist()), exclusive=True, axis=1)
            R_next[j - 1, 0] = (M / jf) * tf.cumsum(tf.add_n(R[j - 2, :].tolist()), exclusive=True, axis=-1)

        # interior
        for j in range(2, d + 1):
            jf = tf.cast(j, _tf_float())
            for k in range(2, d + 1):
                kf = tf.cast(k, _tf_float())
                R_next[j - 1, k - 1] = (M / (jf * kf)) * R[j - 2, k - 2]

        K.append(tf.reduce_sum(tf.add_n(R_next.flatten().tolist()), axis=(1, -1)))
        R = R_next

    return tf.stack(K, axis=0)


# ----------------------------------------------------------------------------- #
# Tensor kernel (dense)
# ----------------------------------------------------------------------------- #
def tensor_kern(M: tf.Tensor, num_levels: int) -> tf.Tensor:
    """
    Inner products of inducing tensors, per signature level.

    Parameters
    ----------
    M : tf.Tensor
        Shape (Ltri, M, M) with Ltri = num_levels*(num_levels+1)/2
    num_levels : int
        Max level

    Returns
    -------
    K_lvls : tf.Tensor
        Shape (num_levels+1, M, M)
    """
    num_tensors = tf.shape(M)[1]
    num_tensors2 = tf.shape(M)[2]

    # level-0 is ones
    K0 = _one_like((num_tensors, num_tensors2), ref_dtype=M.dtype)
    K = [K0]

    k = 0
    for i in range(1, num_levels + 1):
        R = M[k]
        k += 1
        for _ in range(1, i):
            R = M[k] * R
            k += 1
        K.append(R)

    return tf.stack(K, axis=0)


# ----------------------------------------------------------------------------- #
# Tensor vs sequence kernels (dense)
# ----------------------------------------------------------------------------- #
def signature_kern_tens_vs_seq_first_order(
    M: tf.Tensor, num_levels: int, difference: bool = True
) -> tf.Tensor:
    """
    Tensor vs sequence inner products (first order).

    M : (Ltri, num_tensors, num_examples, len_examples)
    Returns: (num_levels+1, num_tensors, num_examples)
    """
    num_tensors, num_examples, _ = tf.unstack(tf.shape(M)[1:])

    if difference:
        M = M[..., 1:] - M[..., :-1]

    # level-0
    K0 = _one_like((num_tensors, num_examples), ref_dtype=M.dtype)
    K = [K0]

    k = 0
    for i in range(1, num_levels + 1):
        R = M[k]; k += 1
        for _ in range(1, i):
            R = M[k] * tf.cumsum(R, exclusive=True, axis=2); k += 1
        K.append(tf.reduce_sum(R, axis=2))

    return tf.stack(K, axis=0)


def signature_kern_tens_vs_seq_higher_order(
    M: tf.Tensor, num_levels: int, order: int = 2, difference: bool = True
) -> tf.Tensor:
    """
    Tensor vs sequence inner products (higher order).

    M : (Ltri, num_tensors, num_examples, len_examples)
    Returns: (num_levels+1, num_tensors, num_examples)
    """
    num_tensors, num_examples, _ = tf.unstack(tf.shape(M)[1:])

    if difference:
        M = M[..., 1:] - M[..., :-1]

    # level-0
    K0 = _one_like((num_tensors, num_examples), ref_dtype=M.dtype)
    K = [K0]

    k = 0
    for i in range(1, num_levels + 1):
        R = np.asarray([M[k]]); k += 1
        for j in range(1, i):
            d = min(j + 1, order)
            R_next = np.empty((d,), dtype=object)
            R_next[0] = M[k] * tf.cumsum(tf.add_n(R.tolist()), exclusive=True, axis=2)
            for l in range(1, d):
                denom = tf.cast(l + 1, _tf_float())
                R_next[l] = (M[k] / denom) * R[l - 1]
            R = R_next; k += 1
        K.append(tf.reduce_sum(tf.add_n(R.tolist()), axis=2))

    return tf.stack(K, axis=0)


# ----------------------------------------------------------------------------- #
# Low-rank (feature) versions
# ----------------------------------------------------------------------------- #
def signature_kern_first_order_lr_feature(
    U: tf.Tensor,
    num_levels: int,
    rank_bound: int,
    sparsity: str = "sqrt",
    seeds: tf.Tensor | None = None,
    difference: bool = True,
):
    """
    Low-rank feature maps for first-order signatures.

    U : (N, L, R)  low-rank features of sequences (post-Nystrom)
    Returns: list length (num_levels+1) with shapes:
      Phi[0] : (N, 1)   (all ones)
      Phi[i] : (N, r_i) for i >= 1
    """
    N = tf.shape(U)[0]

    # level-0: ones
    Phi0 = _one_like((N, 1), ref_dtype=U.dtype)
    Phi = [Phi0]

    V = U
    if difference:
        V = V[:, 1:, :] - V[:, :-1, :]

    # level-1: sum over time
    Phi.append(tf.reduce_sum(V, axis=1))

    P = V
    for i in range(2, num_levels + 1):
        P = tf.cumsum(P, axis=1, exclusive=True)
        if seeds is None:
            P = lr_hadamard_prod_rand(V, P, rank_bound, sparsity)
        else:
            P = lr_hadamard_prod_rand(V, P, rank_bound, sparsity, seeds[i - 2])
        # Per your original code, append sum(V) at each level
        Phi.append(tf.reduce_sum(V, axis=1))

    return Phi


def tensor_kern_lr_feature(
    U: tf.Tensor,
    num_levels: int,
    rank_bound: int,
    sparsity: str = "sqrt",
    seeds: tf.Tensor | None = None,
):
    """
    Low-rank feature maps for inducing tensors.

    U : (Ltri, M, R)
    Returns: list length (num_levels+1):
      Phi[0] : (M, 1) (ones)
      Phi[i] : (M, r_i) for i >= 1
    """
    # level-0: ones
    zeros = tf.reduce_sum(U[0] * 0.0, axis=1, keepdims=True)
    Phi = [zeros + tf.cast(1.0, _tf_float())]

    k = 0
    for i in range(1, num_levels + 1):
        R = U[k]; k += 1
        for j in range(1, i):
            if seeds is None:
                R = lr_hadamard_prod_rand(U[k], R, rank_bound, sparsity)
            else:
                R = lr_hadamard_prod_rand(U[k], R, rank_bound, sparsity, seeds[j - 1])
            k += 1
        Phi.append(R)
    return Phi
