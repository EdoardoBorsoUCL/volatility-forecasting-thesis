from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import gpflow as gf

from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.covariances import Kuu, Kuf  # GPflow 2 singledispatch

# Import your (ported) kernels; keep the same class names & methods as in TF1
from kernels import SignatureKernel, SignatureRBF

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _tf_float() -> tf.DType:
    return default_float()

def _to_float(x):
    f = _tf_float()
    x = tf.convert_to_tensor(x)
    return tf.cast(x, f) if x.dtype != f else x


# -----------------------------------------------------------------------------
# Base class (keeps the same semantics as the TF1 version)
# -----------------------------------------------------------------------------
class SignatureInducing(gf.inducing_variables.InducingVariables):
    """
    Base class for inducing variables for signature kernels in sparse variational GPs.

    Parameters
    ----------
    Z : array-like
        Locations of inducing variables. Subclasses define shape.
    num_levels : int
        Same truncation level as in SignatureKernel.
    learn_weights : bool
        If True, add a per-level linear combination W (shape [num_levels, M, M]).
    """
    def __init__(self, Z, num_levels: int, learn_weights: bool = False):
        super().__init__()
        self.learn_weights = bool(learn_weights)
        self.num_levels = int(num_levels)

        Z = _to_float(Z)
        # In GPflow 2, Parameter wraps a tf.Variable and is already "as tensor"
        self.Z = Parameter(Z)

        if self.learn_weights:
            # Initialize W as per-level identity, shape [num_levels, M, M]
            M = int(self.__len__())
            eye = tf.eye(M, dtype=_tf_float())
            W0 = tf.repeat(eye[None, ...], repeats=self.num_levels, axis=0)
            self.W = Parameter(W0)  # no transform to match the TF1 behavior
        else:
            self.W = None

    def __len__(self) -> int:
        # Subclasses must implement. It depends on how Z is laid out.
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Inducing TENSORS (same layout & semantics as original)
# -----------------------------------------------------------------------------
class InducingTensors(SignatureInducing):
    """
    Inducing class for using sparse tensors as inducing variable locations.

    Inputs
    ------
    Z : np.ndarray
        If not 'increments':
          shape = [ (L+1)*L/2, M, D ]  (L := num_levels)
        If 'increments' is True:
          shape = [ (L+1)*L/2, M, 2, D ]
    num_levels : int
        Same as SignatureKernel.num_levels
    increments : bool
        Whether each axis in the tensor product uses differences of RK (two entries).
    """
    def __init__(self, Z, num_levels: int, increments: bool = False, **kwargs):
        Z = np.asarray(Z)
        len_tensors = int(num_levels * (num_levels + 1) / 2)
        if Z.shape[0] != len_tensors:
            raise ValueError(f"InducingTensors: Z.shape[0] must be Ltri={len_tensors}, got {Z.shape[0]}")
        if increments:
            if Z.ndim != 4 or Z.shape[2] != 2:
                raise ValueError("increments=True expects Z shape [Ltri, M, 2, D]")

        super().__init__(Z, num_levels, **kwargs)
        self.len_tensors = len_tensors
        self.increments = bool(increments)

    def __len__(self) -> int:
        # number of tensor "atoms"
        return int(self.Z.shape[1])
    
     # For GPflow shape check sanity (optional but helpful)
    @property
    def num_inducing(self) -> int:
        return self.__len__()

    @property
    def shape(self) -> Tuple[int]:
        # Posterior creation sometimes inspects this
        return (int(self.Z.shape[0]),)


# -----------------------------------------------------------------------------
# Inducing SEQUENCES (same semantics as original)
# -----------------------------------------------------------------------------
class InducingSequences(SignatureInducing):
    """
    Inducing class for using sequences as inducing variable locations.

    Inputs
    ------
    Z : np.ndarray
        shape = [M, L, d]  (num_inducing, len_inducing, num_features)
    num_levels : int
        Same as SignatureKernel.num_levels
    """
    def __init__(self, Z, num_levels: int, **kwargs):
        Z = np.asarray(Z)
        if Z.ndim != 3:
            raise ValueError("InducingSequences expects Z with shape [M, L, d].")
        super().__init__(Z, num_levels, **kwargs)
        self.len_inducing = int(Z.shape[1])  # L

    def __len__(self) -> int:
        # number of inducing sequences
        return int(self.Z.shape[0])

    # For GPflow shape check sanity (optional but helpful)
    @property
    def num_inducing(self) -> int:
        return self.__len__()

    @property
    def shape(self) -> Tuple[int]:
        # Posterior creation sometimes inspects this
        return (int(self.Z.shape[0]),)



# ===== InducingTensors × SignatureKernel =====================================

@Kuu.register(InducingTensors, SignatureKernel)
def Kuu_indtens_sigkernel(
    feat: InducingTensors,
    kern: SignatureKernel,
    *,
    jitter: Optional[tf.Tensor] = None,
    full_f_cov: bool = False,   # ignored here (matches original signature)
) -> tf.Tensor:
    """
    Kuu for tensor-inducing variables under SignatureKernel.
    Respects 'learn_weights' and 'increments' as in the TF1 version.
    """
    Z = _to_float(feat.Z)

    if feat.learn_weights:
        # Return per-level [L+1, M, M], combine with W the higher levels
        K_lvls = kern.K_tens(Z, return_levels=True, increments=feat.increments)
        K0 = K_lvls[0]
        higher = K_lvls[1:]
        W = _to_float(feat.W)  # [L, M, M]
        K = K0 + tf.add_n([W[i] @ higher[i] @ tf.transpose(W[i]) for i in range(kern.num_levels)])
    else:
        K = kern.K_tens(Z, increments=feat.increments)  # [M, M]

    j = default_jitter() if jitter is None else tf.cast(jitter, _tf_float())
    M = tf.shape(K)[0]
    return K + j * tf.eye(M, dtype=_tf_float())


@Kuf.register(InducingTensors, SignatureKernel, tf.Tensor)
def Kuf_indtens_sigkernel(
    feat: InducingTensors,
    kern: SignatureKernel,
    X_new: tf.Tensor,
) -> tf.Tensor:
    """
    Kuf for tensor-inducing vs new *sequences* under SignatureKernel.
    Matches the original TF1 code (learn_weights controls per-level mixing on the Z side).
    """
    Z = _to_float(feat.Z)
    X_new = _to_float(X_new)

    if feat.learn_weights:
        K_lvls = kern.K_tens_vs_seq(Z, X_new, return_levels=True, increments=feat.increments)
        K0 = K_lvls[0]
        higher = K_lvls[1:]
        W = _to_float(feat.W)  # [L, M, M]
        return K0 + tf.add_n([W[i] @ higher[i] for i in range(kern.num_levels)])
    else:
        return kern.K_tens_vs_seq(Z, X_new, increments=feat.increments)


# Convenience triple (not part of GPflow’s singledispatch; handy if your code calls it)
def Kuu_Kuf_Kff_tensors(
    feat: InducingTensors,
    kern: SignatureKernel,
    X_new: tf.Tensor,
    *,
    jitter: float | tf.Tensor = 0.0,
    full_f_cov: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Exact TF2 port of the TF1 helper:
      - If learn_weights, combine per-level with W for Kzz and Kzx; sum levels for Kxx
      - Add jitter to Kzz and (optionally) Kxx diagonals
    """
    Z = _to_float(feat.Z)
    X_new = _to_float(X_new)

    if feat.learn_weights:
        Kzz_lvls, Kzx_lvls, Kxx_lvls = kern.K_tens_n_seq_covs(
            Z, X_new, full_X_cov=full_f_cov, return_levels=True, increments=feat.increments
        )
        W = _to_float(feat.W)  # [L, M, M]
        Kzz = Kzz_lvls[0] + tf.add_n([W[i] @ Kzz_lvls[1:][i] @ tf.transpose(W[i]) for i in range(kern.num_levels)])
        Kzx = Kzx_lvls[0] + tf.add_n([W[i] @ Kzx_lvls[1:][i] for i in range(kern.num_levels)])
        Kxx = tf.reduce_sum(Kxx_lvls, axis=0)
    else:
        Kzz, Kzx, Kxx = kern.K_tens_n_seq_covs(
            Z, X_new, full_X_cov=full_f_cov, increments=feat.increments
        )

    j = tf.cast(jitter, _tf_float())
    M = tf.shape(Kzz)[0]
    Kzz = Kzz + j * tf.eye(M, dtype=_tf_float())
    if full_f_cov:
        N = tf.shape(X_new)[0]
        Kxx = Kxx + j * tf.eye(N, dtype=_tf_float())
    else:
        Kxx = Kxx + j
    return Kzz, Kzx, Kxx


# ===== InducingSequences × SignatureKernel ===================================

@Kuu.register(InducingSequences, SignatureKernel)
def Kuu_indseq_sigkernel(
    feat: InducingSequences,
    kern: SignatureKernel,
    *,
    jitter: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    """
    Kuu for sequence-inducing variables under SignatureKernel.
    Uses the same "presliced=True" convention as TF1.
    """
    Z = _to_float(feat.Z)  # [M, L, d]

    if feat.learn_weights:
        # Per-level [L+1, M, M], mix higher levels by W
        K_lvls = kern.K(Z, return_levels=True, presliced=True)  # kernel must support this kwarg
        K0 = K_lvls[0]
        higher = K_lvls[1:]
        W = _to_float(feat.W)  # [L, M, M]
        Kzz = K0 + tf.add_n([W[i] @ higher[i] @ tf.transpose(W[i]) for i in range(kern.num_levels)])
    else:
        Kzz = kern.K(Z, presliced=True)  # summed across levels

    j = default_jitter() if jitter is None else tf.cast(jitter, _tf_float())
    M = tf.shape(Z)[0]
    return Kzz + j * tf.eye(M, dtype=_tf_float())


@Kuf.register(InducingSequences, SignatureKernel, tf.Tensor)
def Kuf_indseq_sigkernel(
    feat: InducingSequences,
    kern: SignatureKernel,
    X_new: tf.Tensor,
) -> tf.Tensor:
    """
    Kuf for sequence-inducing vs new sequences under SignatureKernel.
    Matches TF1: if learn_weights, mix levels with W on the Z side.
    """
    Z = _to_float(feat.Z)      # [M, L, d]
    X_new = _to_float(X_new)   # usually [N, L, d]; if flat, kernel should handle presliced_X=False by default

    if feat.learn_weights:
        K_lvls = kern.K(Z, X_new, presliced_X=True, return_levels=True)  # kernel must support these kwargs
        K0 = K_lvls[0]
        higher = K_lvls[1:]
        W = _to_float(feat.W)  # [L, M, M]
        return K0 + tf.add_n([W[i] @ higher[i] for i in range(kern.num_levels)])
    else:
        return kern.K(Z, X_new, presliced_X=True)


def Kuu_Kuf_Kff_sequences(
    feat: InducingSequences,
    kern: SignatureKernel,
    X_new: tf.Tensor,
    *,
    jitter: float | tf.Tensor = 0.0,
    full_f_cov: bool = False,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    TF2 port of the TF1 helper for sequences:
      - If learn_weights, combine per-level with W for Kzz and Kzx; sum levels for Kxx
      - Add jitter to Kzz and (optionally) Kxx diagonals
    """
    Z = _to_float(feat.Z)
    X_new = _to_float(X_new)

    if feat.learn_weights:
        Kzz_lvls, Kzx_lvls, Kxx_lvls = kern.K_seq_n_seq_covs(
            Z, X_new, full_X2_cov=full_f_cov, return_levels=True
        )
        W = _to_float(feat.W)  # [L, M, M]
        Kzz = Kzz_lvls[0] + tf.add_n([W[i] @ Kzz_lvls[1:][i] @ tf.transpose(W[i]) for i in range(kern.num_levels)])
        Kzx = Kzx_lvls[0] + tf.add_n([W[i] @ Kzx_lvls[1:][i] for i in range(kern.num_levels)])
        Kxx = tf.reduce_sum(Kxx_lvls, axis=0)
    else:
        Kzz, Kzx, Kxx = kern.K_seq_n_seq_covs(Z, X_new, full_X2_cov=full_f_cov)

    j = tf.cast(jitter, _tf_float())
    M = tf.shape(Z)[0]
    Kzz = Kzz + j * tf.eye(M, dtype=_tf_float())
    if full_f_cov:
        N = tf.shape(X_new)[0]
        Kxx = Kxx + j * tf.eye(N, dtype=_tf_float())
    else:
        Kxx = Kxx + j
    return Kzz, Kzx, Kxx
