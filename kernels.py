# kernels.py  — TF2 / GPflow 2 port (fixed active_dims slicing across time & lags)

from __future__ import annotations
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
import gpflow as gf

from gpflow.base import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel
from gpflow.utilities import positive
try:
    # GPflow 2.x provides a bounded sigmoid bijector
    from gpflow.utilities.bijectors import Sigmoid
except Exception:
    Sigmoid = None  # fallback if not available

# TF2 versions of your modules (same APIs as TF1 versions)
import lags
import low_rank_calculations
import signature_algs


# -----------------------------------------------------------------------------#
# Small dtype helpers
# -----------------------------------------------------------------------------#
def _tf_float() -> tf.DType:
    return default_float()

def _to_float(x) -> tf.Tensor:
    x = tf.convert_to_tensor(x)
    return tf.cast(x, _tf_float()) if x.dtype != _tf_float() else x

def _draw_indices_tf(total_count, k, seed=None):
    total_count = tf.cast(total_count, tf.int32)
    k = tf.cast(k, tf.int32)
    if seed is None:
        perm = tf.random.shuffle(tf.range(total_count))
    else:
        g = tf.random.Generator.from_seed(int(seed))
        perm = g.permutation(tf.range(total_count))
    return perm[:k]


# -----------------------------------------------------------------------------#
# Core Signature Kernel
# -----------------------------------------------------------------------------#
class SignatureKernel(Kernel):
    """
    TF2/GPflow2 port of the SignatureKernel with:
      - normalization
      - difference (integrated path vs increments)
      - low_rank mode (Nyström + level features)
      - lags and per-dimension lengthscales
      - per-level variances and global sigma
      - presliced/presliced_X/presliced_X2/return_levels flags

    IMPORTANT CHANGE:
    active_dims now slices **per-feature across all time steps (and lags)**,
    not on the flattened (L*d) axis, to avoid throwing away most timesteps.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        num_levels: int,
        active_dims: Optional[List[int]] = None,
        variances=1,
        lengthscales=1,
        order: int = 1,
        normalization: bool = True,
        difference: bool = True,
        num_lags: Optional[int] = None,
        low_rank: bool = False,
        num_components: int = 50,
        rank_bound: Optional[int] = None,
        sparsity: str = "sqrt",
        name: Optional[str] = None,
    ):
        super().__init__(active_dims=active_dims, name=name)

        self.input_dim    = int(input_dim)
        self.num_features = int(num_features)
        self.num_levels   = int(num_levels)
        self.len_examples = self._validate_number_of_features(self.input_dim, self.num_features)

        # order in [1, num_levels] ; -1 or >= num_levels -> num_levels
        self.order = self.num_levels if (order <= 0 or order >= self.num_levels) else int(order)
        if self.order != 1 and low_rank:
            raise NotImplementedError("Higher-order algorithms not compatible with low-rank mode (yet).")

        self.normalization = bool(normalization)
        self.difference    = bool(difference)

        # Per-level variances (size L+1), and global sigma
        var_init = self._validate_signature_param("variances", variances, self.num_levels + 1)
        self.variances = Parameter(_to_float(var_init), transform=positive())
        self.sigma     = Parameter(_to_float(1.0),      transform=positive())

        # Low-rank controls
        self.low_rank, self.num_components, self.rank_bound, self.sparsity = \
            self._validate_low_rank_params(low_rank, num_components, rank_bound, sparsity)

        # Lags & gamma weights
        if num_lags is None:
            self.num_lags = 0
        else:
            if not isinstance(num_lags, int) or num_lags < 0:
                raise ValueError("num_lags must be a nonnegative integer or None.")
            self.num_lags = int(num_lags)
            if self.num_lags > 0:
                # In GPflow1: transforms.Logistic() -> (0,1). Use bounded Sigmoid if available, else Positive.
                lags_init = 0.1 * np.asarray(range(1, self.num_lags + 1), dtype=np.float64)
                if Sigmoid is not None:
                    self.lags  = Parameter(_to_float(lags_init), transform=Sigmoid(lower=0.0, upper=1.0))
                else:
                    self.lags  = Parameter(_to_float(lags_init), transform=positive())  # fallback
                gamma = 1.0 / np.asarray(range(1, self.num_lags + 2), dtype=np.float64)
                gamma /= np.sum(gamma)
                self.gamma = Parameter(_to_float(gamma), transform=positive())

        # Lengthscales per feature (with a small lower bound to avoid collapse)
        if lengthscales is not None:
            ls_init = self._validate_signature_param("lengthscales", lengthscales, self.num_features)
            self.lengthscales = Parameter(_to_float(ls_init), transform=positive(lower=1e-3))
        else:
            self.lengthscales = None

        # Base kernel pointer (set by subclasses)
        self._base_kern = None  # will be assigned in subclasses

    # ------------------------------ Validators ------------------------------ #
    def _validate_number_of_features(self, input_dim, num_features) -> int:
        if input_dim % num_features == 0:
            return int(input_dim // num_features)
        raise ValueError("The arguments num_features and input_dim are not consistent.")

    def _validate_low_rank_params(self, low_rank, num_components, rank_bound, sparsity):
        if low_rank is not None and low_rank is True:
            if not isinstance(low_rank, bool):
                raise ValueError("low_rank must be True/False.")
            if sparsity not in ["log", "sqrt", "lin"]:
                raise ValueError("Unknown sparsity '%s'. Use 'sqrt', 'log', or 'lin'." % sparsity)
            if rank_bound is not None and rank_bound <= 0:
                raise ValueError("rank_bound must be None or a positive integer.")
            if num_components is None or num_components <= 0:
                raise ValueError("num_components must be a positive integer.")
            if rank_bound is None:
                rank_bound = num_components
        else:
            low_rank = False
        return bool(low_rank), int(num_components), (None if rank_bound is None else int(rank_bound)), sparsity

    def _validate_signature_param(self, name, value, length):
        value = np.asarray(value, dtype=np.float64) * np.ones(length, dtype=np.float64)
        correct_shape = () if length == 1 else (length,)
        if np.asarray(value).squeeze().shape != correct_shape:
            raise ValueError(f"shape of parameter {name} is not what is expected ({length})")
        return value

    # ------------------------------ Active-dims slicing (FIXED) ------------- #
    def _expand_feature_indices_across_time(self):
        """
        Expand per-feature active_dims (0..d-1) to indices on the last feature axis
        valid per time step and across lags. Supports list/ndarray OR slice.
        Returns:
            idx_step: shape (lag_mult * |active_dims|,) or None for no-op.
        """
        ad = self.active_dims
        if ad is None:
            return None

        # Handle slice(None) → no-op
        if isinstance(ad, slice):
            if ad.start is None and ad.stop is None and ad.step is None:
                return None  # full slice → keep all features
            # Concrete slice → build integer indices
            start = 0 if ad.start is None else int(ad.start)
            stop  = self.num_features if ad.stop is None else int(ad.stop)
            step  = 1 if ad.step is None else int(ad.step)
            idx_feat = tf.range(start, stop, delta=step, dtype=tf.int32)
        else:
            # list/tuple/ndarray
            idx_feat = tf.convert_to_tensor(ad, dtype=tf.int32)

        # replicate indices across lag blocks: [idx + k*d for k in 0..num_lags]
        offs = tf.range(self.num_lags + 1, dtype=tf.int32) * self.num_features
        idx_step = tf.reshape(idx_feat[None, :] + offs[:, None], [-1])
        return idx_step


    def _maybe_slice(self, X, X2):
        """
        Slice by feature indices across **all time steps and lags**.
        Works when X is (N, L*d_eff) or already flat; we reshape to (N,L,d_eff),
        gather feature dims, then flatten back.
        """
        if self.active_dims is None:
            return X, X2
        idx_step = self._expand_feature_indices_across_time()
        if idx_step is None:
            return X, X2

        def slice_tensor(T):
            if T is None:
                return None
            N = tf.shape(T)[0]
            # total features per time step (after adding lags)
            d_eff = self.num_features * (self.num_lags + 1)
            X3 = tf.reshape(T, [N, -1, d_eff])             # (N, L, d_eff)
            X3s = tf.gather(X3, idx_step, axis=-1)         # (N, L, |idx_step|)
            return tf.reshape(X3s, [N, -1])                # (N, L*|idx_step|)

        return slice_tensor(X), slice_tensor(X2)

    # ------------------------------ Public helpers (former @autoflow) ------- #
    # Kept for backward-compatibility with your notebook calls.
    def compute_K(self, X, Y):
        return self.K(X, Y)

    def compute_K_symm(self, X):
        return self.K(X)

    def compute_base_kern_symm(self, X):
        num_examples = tf.shape(X)[0]
        X = tf.reshape(X, (num_examples, -1, self.num_features))
        len_examples = tf.shape(X)[1]
        Xflat = tf.reshape(self._apply_scaling_and_lags_to_sequences(X), (-1, self.num_features))
        M = tf.transpose(
            tf.reshape(self._base_kern(Xflat), [num_examples, len_examples, num_examples, len_examples]),
            [0, 2, 1, 3],
        )
        return M

    def compute_K_level_diags(self, X):
        return self.K_diag(X, return_levels=True)

    def compute_K_levels(self, X, X2):
        return self.K(X, X2, return_levels=True)

    def compute_K_diag(self, X):
        return self.K_diag(X)

    def compute_K_tens(self, Z):
        return self.K_tens(Z, return_levels=False)

    def compute_K_tens_vs_seq(self, Z, X):
        return self.K_tens_vs_seq(Z, X, return_levels=False)

    def compute_K_incr_tens(self, Z):
        return self.K_tens(Z, increments=True, return_levels=False)

    def compute_K_incr_tens_vs_seq(self, Z, X):
        return self.K_tens_vs_seq(Z, X, increments=True, return_levels=False)

    # ------------------------------ Internals: sequence covariance (dense) --- #
    def _K_seq_diag(self, X):
        """
        X: (N, L, d)
        returns: (L+1, N) — per-example diagonals at each signature level
        """
        N = tf.shape(X)[0]
        L = tf.shape(X)[1]
        d = tf.shape(X)[2]

        # Build full Gram on flattened time, then extract the N diagonal (L x L) blocks
        Xflat = tf.reshape(X, [N * L, d])                           # (N*L, d)
        G = self._base_kern(Xflat, Xflat)                           # (N*L, N*L)
        G4 = tf.reshape(G, [N, L, N, L])                            # (N, L, N, L)
        G4 = tf.transpose(G4, [0, 2, 1, 3])                         # (N, N, L, L)
        ii = tf.range(N, dtype=tf.int32)
        M3 = tf.gather_nd(G4, tf.stack([ii, ii], axis=1))           # (N, L, L)

        # Now feed (N, L, L) into signature_algs; it returns (L+1, N)
        if self.order == 1:
            K_lvls_diag = signature_algs.signature_kern_first_order(
                M3, self.num_levels, difference=self.difference
            )                                                       # (L+1, N)
        else:
            K_lvls_diag = signature_algs.signature_kern_higher_order(
                M3, self.num_levels, order=self.order, difference=self.difference
            )                                                       # (L+1, N)

        return K_lvls_diag

    def _K_seq(self, X, X2=None):
        """
        X : (N, L, d)
        X2: (N2, L2, d) or None
        returns levels Gram: (L+1, N, N2)
        """
        N  = tf.shape(X)[0]
        L  = tf.shape(X)[1]
        d  = tf.shape(X)[2]
        NL = N * L

        if X2 is None:
            Xflat = tf.reshape(X, [NL, d])
            M = tf.reshape(self._base_kern(Xflat), [N, L, N, L])
        else:
            N2  = tf.shape(X2)[0]
            L2  = tf.shape(X2)[1]
            N2L2 = N2 * L2
            Xflat  = tf.reshape(X,  [NL, d])
            X2flat = tf.reshape(X2, [N2L2, d])
            M = tf.reshape(self._base_kern(Xflat, X2flat), [N, L, N2, L2])

        if self.order == 1:
            K_lvls = signature_algs.signature_kern_first_order(M, self.num_levels, difference=self.difference)
        else:
            K_lvls = signature_algs.signature_kern_higher_order(M, self.num_levels, order=self.order, difference=self.difference)
        return K_lvls

    # ------------------------------ Internals: sequence features (low-rank) -- #
    def _K_seq_lr_feat(self, X, nys_samples=None, seeds=None):
        """
        X: (N, L, d)  -> returns list length (L+1) of [N, r_l] factors
        """
        N = tf.shape(X)[-3]
        L = tf.shape(X)[-2]
        d = tf.shape(X)[-1]
        NL = N * L

        Xflat = tf.reshape(X, [NL, d])
        X_feat = low_rank_calculations.Nystrom_map(Xflat, self._base_kern, nys_samples, self.num_components)
        X_feat = tf.reshape(X_feat, [N, L, self.num_components])

        if self.order == 1:
            Phi_lvls = signature_algs.signature_kern_first_order_lr_feature(
                X_feat, self.num_levels, self.rank_bound, self.sparsity, seeds, difference=self.difference
            )
        else:
            raise NotImplementedError("Low-rank mode not implemented for order > 1.")
        return Phi_lvls

    # ------------------------------ Internals: tensor covariances ----------- #
    def _K_tens(self, Z, increments=False):
        """
        Z: [Ltri, M, D] or [Ltri, M, 2, D] if increments
        returns per-level list/stack: (L+1, M, M)
        """
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]
        D    = tf.shape(Z)[-1]

        if increments:
            Z2  = tf.reshape(Z, [Ltri, 2 * M, D])
            Mbl = tf.reshape(self._base_kern(Z2), [Ltri, M, 2, M, 2])
            Mbl = Mbl[:, :, 1, :, 1] + Mbl[:, :, 0, :, 0] - Mbl[:, :, 1, :, 0] - Mbl[:, :, 0, :, 1]
        else:
            Mbl = self._base_kern(Z)

        K_lvls = signature_algs.tensor_kern(Mbl, self.num_levels)
        return K_lvls

    def _K_tens_lr_feat(self, Z, increments=False, nys_samples=None, seeds=None):
        """
        Returns list of per-level factors for tensor-inducing covariances.
        """
        if self.order > 1:
            raise NotImplementedError("higher order not implemented yet for low-rank mode")

        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]
        D    = tf.shape(Z)[-1]

        if increments:
            Zflat  = tf.reshape(Z, [M * Ltri * 2, D])
            Z_feat = low_rank_calculations.Nystrom_map(Zflat, self._base_kern, nys_samples, self.num_components)
            Z_feat = tf.reshape(Z_feat, [Ltri, M, 2, self.num_components])
            Z_feat = Z_feat[:, :, 1, :] - Z_feat[:, :, 0, :]
        else:
            Zflat  = tf.reshape(Z, [M * Ltri, D])
            Z_feat = low_rank_calculations.Nystrom_map(Zflat, self._base_kern, nys_samples, self.num_components)
            Z_feat = tf.reshape(Z_feat, [Ltri, M, self.num_components])

        Phi_lvls = signature_algs.tensor_kern_lr_feature(
            Z_feat, self.num_levels, self.rank_bound, self.sparsity, seeds
        )
        return Phi_lvls

    def _K_tens_vs_seq(self, Z, X, increments: bool = False):
        """
        Dense tensor-vs-sequence covariance.
        Z : tensors     (Ltri, M, d)          or (Ltri, M, 2, d) if increments=True
        X : sequences   (N, L, d)
        returns         list/stack per level: (L+1, M, N)
        """
        # shapes
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]
        D    = tf.shape(Z)[-1]
        N    = tf.shape(X)[-3]
        L    = tf.shape(X)[-2]

        # flatten sequences: (N*L, d)
        Xflat = tf.reshape(X, [N * L, tf.shape(X)[-1]])

        # Choose pairwise routine:
        pairwise = getattr(self, "_spectral_blocks", None)  # present only on SignatureSpectral

        if increments:
            Zflat = tf.reshape(Z, [2 * Ltri * M, D])  # (2*M*Ltri, d)
            Kfull = pairwise(Zflat, Xflat) if pairwise is not None else self._base_kern(Zflat, Xflat)
            Mbl = tf.reshape(Kfull, (Ltri, M, 2, N, L))   # (Ltri, M, 2, N, L)
            Mbl = Mbl[:, :, 1] - Mbl[:, :, 0]             # (Ltri, M, N, L)
        else:
            Zflat = tf.reshape(Z, [Ltri * M, D])          # (M*Ltri, d)
            Kfull = pairwise(Zflat, Xflat) if pairwise is not None else self._base_kern(Zflat, Xflat)
            Mbl = tf.reshape(Kfull, (Ltri, M, N, L))      # (Ltri, M, N, L)

        # Combine per-level signature blocks → (L+1, M, N)
        if self.order == 1:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_first_order(
                Mbl, self.num_levels, difference=self.difference
            )
        else:
            K_lvls = signature_algs.signature_kern_tens_vs_seq_higher_order(
                Mbl, self.num_levels, order=self.order, difference=self.difference
            )
        return K_lvls

    # ------------------------------ Scaling & Lags ------------------------------ #
    def _apply_scaling_and_lags_to_sequences(self, X):
        """
        X: (N, L, d) -> apply lags, lengthscales, gamma weights; returns (N, L, d*(num_lags+1))
        """
        N = tf.shape(X)[0]
        L = tf.shape(X)[1]

        num_features_eff = self.num_features * (self.num_lags + 1)

        Y = X
        if self.num_lags > 0:
            # lags.add_lags_to_sequences should be TF2-compatible
            Y = lags.add_lags_to_sequences(Y, self.lags)

        # shape: (N, L, num_lags+1, num_features)
        Y = tf.reshape(Y, (N, L, self.num_lags + 1, self.num_features))

        if self.lengthscales is not None:
            Y = Y / self.lengthscales[None, None, None, :]

        if self.num_lags > 0:
            Y = Y * self.gamma[None, None, :, None]

        Y = tf.reshape(Y, (N, L, num_features_eff))
        return Y

    def _apply_scaling_to_tensors(self, Z):
        """
        Z simple tensors: (Ltri, M, d*(num_lags+1))
        """
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]
        if self.lengthscales is not None:
            Z4 = tf.reshape(Z, (Ltri, M, self.num_lags + 1, self.num_features))
            Z4 = Z4 / self.lengthscales[None, None, None, :]
            if self.num_lags > 0:
                Z4 = Z4 * self.gamma[None, None, :, None]
            Z  = tf.reshape(Z4, (Ltri, M, -1))
        return Z

    def _apply_scaling_to_incremental_tensors(self, Z):
        """
        Z incremental tensors: (Ltri, M, 2, d*(num_lags+1))
        """
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]
        D    = tf.shape(Z)[-1]

        if self.lengthscales is not None:
            Z5 = tf.reshape(Z, (Ltri, M, 2, self.num_lags + 1, self.num_features))
            Z5 = Z5 / self.lengthscales[None, None, None, None, :]
            if self.num_lags > 0:
                Z5 = Z5 * self.gamma[None, None, None, :, None]
            Z  = tf.reshape(Z5, (Ltri, M, 2, D))
        return Z

    # ------------------------------ Public K / Kdiag & friends ---------------- #
    def K(self, X, X2=None, *, presliced=False, return_levels=False, presliced_X=False, presliced_X2=False):
        """
        Computes signature kernel between sequences.
        Preserves all flags from TF1 version.
        Accepts X in either (N, D) with D=L*d (then we reshape) or already (N, L, d) if presliced_X.
        """
        # Handle slicing by active_dims (if provided)
        if presliced:
            presliced_X = True
            presliced_X2 = True

        if not presliced_X and not presliced_X2:
            X, X2 = self._maybe_slice(X, X2)
        elif not presliced_X:
            X, _  = self._maybe_slice(X, None)
        elif not presliced_X2 and X2 is not None:
            X2, _ = self._maybe_slice(X2, None)

        # Reshape to (N, L, d)
        N  = tf.shape(X)[0]
        X3 = tf.reshape(X, [N, -1, self.num_features])
        L  = tf.shape(X3)[1]

        X_scaled = self._apply_scaling_and_lags_to_sequences(X3)

        if X2 is None:
            if self.low_rank:
                Phi_lvls = self._K_seq_lr_feat(X3)
                K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
            else:
                K_lvls = self._K_seq(X_scaled)

            if self.normalization:
                jit = default_jitter()
                K_lvls = K_lvls + jit * tf.eye(N, dtype=_tf_float())[None]
                K_lvls_diag_sqrt = tf.sqrt(tf.linalg.diag_part(K_lvls))
                K_lvls = K_lvls / (K_lvls_diag_sqrt[:, :, None] * K_lvls_diag_sqrt[:, None, :])
        else:
            N2  = tf.shape(X2)[0]
            X23 = tf.reshape(X2, [N2, -1, self.num_features])
            L2  = tf.shape(X23)[1]
            X2_scaled = self._apply_scaling_and_lags_to_sequences(X23)

            if self.low_rank:
                # seeds and Nystrom samples
                seeds = tf.random.uniform((self.num_levels - 1, 2), minval=0, maxval=2**31 - 1, dtype=tf.int32)
                idx, _ = low_rank_calculations._draw_indices(N * L + N2 * L2, self.num_components)

                nys_samples = tf.gather(
                    tf.concat((tf.reshape(X3, [N * L, -1]), tf.reshape(X23, [N2 * L2, -1])), axis=0),
                    idx,
                    axis=0,
                )

                Phi_lvls  = self._K_seq_lr_feat(X3,  nys_samples=nys_samples, seeds=seeds)
                Phi2_lvls = self._K_seq_lr_feat(X23, nys_samples=nys_samples, seeds=seeds)

                K_lvls = tf.stack(
                    [tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True) for i in range(self.num_levels + 1)],
                    axis=0,
                )
            else:
                K_lvls = self._K_seq(X_scaled, X2_scaled)

            if self.normalization:
                if self.low_rank:
                    K1_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
                    K2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
                else:
                    K1_lvls_diag = self._K_seq_diag(X_scaled)
                    K2_lvls_diag = self._K_seq_diag(X2_scaled)

                jit = default_jitter()
                K1_lvls_diag = K1_lvls_diag + jit
                K2_lvls_diag = K2_lvls_diag + jit

                K1_s = tf.sqrt(K1_lvls_diag)
                K2_s = tf.sqrt(K2_lvls_diag)
                K_lvls = K_lvls / (K1_s[:, :, None] * K2_s[:, None, :])

        # Per-level scaling
        K_lvls = K_lvls * (self.sigma * self.variances[:, None, None])

        if return_levels:
            return K_lvls
        return tf.reduce_sum(K_lvls, axis=0)

    def K_diag(self, X, *, presliced=False, return_levels=False):
        """
        Diagonal of the signature kernel matrix.
        """
        N = tf.shape(X)[0]

        if self.normalization:
            if return_levels:
                return tf.tile((self.sigma * self.variances[:, None]), [1, N])
            else:
                return tf.fill((N,), tf.cast(self.sigma, _tf_float()) * tf.reduce_sum(self.variances))

        if not presliced:
            X, _ = self._maybe_slice(X, None)

        X3 = tf.reshape(X, (N, -1, self.num_features))
        Xs = self._apply_scaling_and_lags_to_sequences(X3)

        if self.low_rank:
            Phi_lvls = self._K_seq_lr_feat(Xs)
            K_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_lvls], axis=0)
        else:
            K_lvls_diag = self._K_seq_diag(Xs)

        K_lvls_diag = K_lvls_diag * (self.sigma * self.variances[:, None])

        if return_levels:
            return K_lvls_diag
        return tf.reduce_sum(K_lvls_diag, axis=0)

    # ------------------------------ Tensor covariances (public) ------------- #
    def K_tens(self, Z, *, return_levels=False, increments=False):
        """
        Square covariance of inducing tensors Z.
        """
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]

        Zs = self._apply_scaling_to_incremental_tensors(Z) if increments else self._apply_scaling_to_tensors(Z)

        if self.low_rank:
            Phi_lvls = self._K_tens_lr_feat(Zs, increments=increments)
            K_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
        else:
            K_lvls = self._K_tens(Zs, increments=increments)

        K_lvls = K_lvls * (self.sigma * self.variances[:, None, None])
        if return_levels:
            return K_lvls
        return tf.reduce_sum(K_lvls, axis=0)

    def K_tens_vs_seq(self, Z, X, *, return_levels=False, increments=False, presliced=False):
        """
        Cross-covariance between inducing tensors Z and sequences X.
        Returns:
        - if return_levels: (L+1, M, N)
        - else: (M, N) collapsed across levels
        """
        if not presliced:
            X, _ = self._maybe_slice(X, None)

        # Reshape X to (N, L, num_features)
        N  = tf.shape(X)[0]
        X3 = tf.reshape(X, (N, -1, self.num_features))
        L  = tf.shape(X3)[1]

        # Z: (Ltri, M, ...), get basic counts
        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]

        # Scale inputs into kernel feature space
        Zs = (self._apply_scaling_to_incremental_tensors(Z)
            if increments else self._apply_scaling_to_tensors(Z))
        Xs = self._apply_scaling_and_lags_to_sequences(X3)

        if self.low_rank:
            # ------- Low-rank Nyström path (pure TF) -------
            # Optional per-level seeds for your LR feature maps
            seeds = tf.random.uniform(
                (self.num_levels - 1, 2), minval=0, maxval=2**31 - 1, dtype=tf.int32
            )

            # Build a sampling pool from flattened Zs and Xs
            inc_factor = 2 if increments else 1
            Zflat = tf.reshape(Zs, [tf.shape(Zs)[0] * tf.shape(Zs)[1] * inc_factor, -1])  # [Ltri*M*(1|2), Dz]
            Xflat = tf.reshape(Xs, [tf.shape(Xs)[0] * tf.shape(Xs)[1], -1])               # [N*L, Dx]
            pool  = tf.concat([Zflat, Xflat], axis=0)                                     # [T, D*]

            total = tf.shape(pool)[0]
            k     = tf.minimum(total, tf.cast(self.num_components, tf.int32))

            # Shuffle-without-replacement via random keys (graph-safe)
            keys = tf.random.uniform(shape=[total], dtype=tf.float32)
            idx  = tf.argsort(keys)[:k]                                                   # [k]
            nys_samples = tf.gather(pool, idx, axis=0)                                    # [k, D*]

            # Low-rank feature maps per level (must be TF-only)
            Phi_Z_lvls = self._K_tens_lr_feat(Zs, increments=increments,
                                            nys_samples=nys_samples, seeds=seeds)
            Phi_X_lvls = self._K_seq_lr_feat(Xs,  nys_samples=nys_samples, seeds=seeds)

            # Level-wise K_{ZX} = Phi_Z @ Phi_X^T  → stack as (L+1, M, N)
            Kzx_lvls = tf.stack(
                [tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True)
                for i in range(self.num_levels + 1)],
                axis=0
            )
        else:
            # ------- Dense path -------
            Kzx_lvls = self._K_tens_vs_seq(Zs, Xs, increments=increments)

        # ------- Normalization -------
        if self.normalization:
            if self.low_rank:
                # diag(K_xx) per level = ||Phi_X||^2
                Kxx_lvls_diag = tf.stack(
                    [tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0
                )  # (L+1, N)
            else:
                Kxx_lvls_diag = self._K_seq_diag(Xs)  # (L+1, N)

            Kxx_lvls_diag = Kxx_lvls_diag + default_jitter()
            Kxx_s = tf.sqrt(Kxx_lvls_diag)           # (L+1, N)
            Kzx_lvls = Kzx_lvls / Kxx_s[:, None, :]  # broadcast over M

        # Amplitude / per-level variances
        Kzx_lvls = Kzx_lvls * (self.sigma * self.variances[:, None, None])

        if return_levels:
            return Kzx_lvls  # (L+1, M, N)
        return tf.reduce_sum(Kzx_lvls, axis=0)  # (M, N)

    def K_tens_n_seq_covs(self, Z, X, *, full_X_cov=False, return_levels=False, increments=False, presliced=False):
        """
        Returns Kzz, Kzx, Kxx (Kxx is diagonal if not full_X_cov).
        """
        if not presliced:
            X, _ = self._maybe_slice(X, None)

        N  = tf.shape(X)[0]
        X3 = tf.reshape(X, (N, -1, self.num_features))
        L  = tf.shape(X3)[1]

        Ltri = tf.shape(Z)[0]
        M    = tf.shape(Z)[1]

        Zs = self._apply_scaling_to_incremental_tensors(Z) if increments else self._apply_scaling_to_tensors(Z)
        Xs = self._apply_scaling_and_lags_to_sequences(X3)

        if self.low_rank:
            seeds = tf.random.uniform((self.num_levels - 1, 2), minval=0, maxval=2**31 - 1, dtype=tf.int32)
            idx, _ = low_rank_calculations._draw_indices(
                tf.shape(Zs)[1] * tf.shape(Zs)[0] * (int(increments) + 1) + N * L, self.num_components
            )
            nys_samples = tf.gather(
                tf.concat(
                    (
                        tf.reshape(Zs, [tf.shape(Zs)[0] * tf.shape(Zs)[1] * (int(increments) + 1), -1]),
                        tf.reshape(Xs, [N * L, -1]),
                    ),
                    axis=0,
                ),
                idx,
                axis=0,
            )

            Phi_Z_lvls = self._K_tens_lr_feat(Zs, increments=increments, nys_samples=nys_samples, seeds=seeds)
            Phi_X_lvls = self._K_seq_lr_feat(Xs, nys_samples=nys_samples, seeds=seeds)

            Kzz_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_Z_lvls], axis=0)
            Kzx_lvls = tf.stack([tf.matmul(Phi_Z_lvls[i], Phi_X_lvls[i], transpose_b=True)
                                 for i in range(self.num_levels + 1)], axis=0)
        else:
            Kzz_lvls = self._K_tens(Zs, increments=increments)
            Kzx_lvls = self._K_tens_vs_seq(Zs, Xs, increments=increments)

        if full_X_cov:
            if self.low_rank:
                Kxx_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_X_lvls], axis=0)
            else:
                Kxx_lvls = self._K_seq(Xs)

            if self.normalization:
                jit = default_jitter()
                Kxx_lvls = Kxx_lvls + jit * tf.eye(N, dtype=_tf_float())[None]
                Kxx_s = tf.sqrt(tf.linalg.diag_part(Kxx_lvls))
                Kxx_lvls = Kxx_lvls / (Kxx_s[:, :, None] * Kxx_s[:, None, :])
                Kzx_lvls = Kzx_lvls / (Kxx_s[:, None, :])

            Kxx_lvls = Kxx_lvls * (self.sigma * self.variances[:, None, None])
            Kzz_lvls = Kzz_lvls * (self.sigma * self.variances[:, None, None])
            Kzx_lvls = Kzx_lvls * (self.sigma * self.variances[:, None, None])

            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls
            else:
                return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls, axis=0)
        else:
            if self.low_rank:
                Kxx_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi_X_lvls], axis=0)
            else:
                Kxx_lvls_diag = self._K_seq_diag(Xs)

            if self.normalization:
                Kxx_lvls_diag = Kxx_lvls_diag + default_jitter()
                Kx_s = tf.sqrt(Kxx_lvls_diag)
                Kzx_lvls = Kzx_lvls / Kx_s[:, None, :]
                # normalized diagonals are sigma*variances
                Kxx_lvls_diag = tf.tile((self.sigma * self.variances[:, None]), [1, N])
            else:
                Kxx_lvls_diag = Kxx_lvls_diag * (self.sigma * self.variances[:, None])

            Kzz_lvls = Kzz_lvls * (self.sigma * self.variances[:, None, None])
            Kzx_lvls = Kzx_lvls * (self.sigma * self.variances[:, None, None])

            if return_levels:
                return Kzz_lvls, Kzx_lvls, Kxx_lvls_diag
            else:
                return tf.reduce_sum(Kzz_lvls, axis=0), tf.reduce_sum(Kzx_lvls, axis=0), tf.reduce_sum(Kxx_lvls_diag, axis=0)

    def K_seq_n_seq_covs(self, X, X2, *, full_X2_cov=False, return_levels=False, presliced=False):
        """
        Returns Kxx, Kxx2, Kx2x2 (Kx2x2 is diagonal if not full_X2_cov).
        """
        if not presliced:
            X2, _ = self._maybe_slice(X2, None)

        N  = tf.shape(X)[0]
        X3 = tf.reshape(X,  (N, -1, self.num_features))
        L  = tf.shape(X3)[1]

        N2  = tf.shape(X2)[0]
        X23 = tf.reshape(X2, (N2, -1, self.num_features))
        L2  = tf.shape(X23)[1]

        Xs  = self._apply_scaling_and_lags_to_sequences(X3)
        X2s = self._apply_scaling_and_lags_to_sequences(X23)

        if self.low_rank:
            seeds = tf.random.uniform((self.num_levels - 1, 2), minval=0, maxval=2**31 - 1, dtype=tf.int32)
            idx, _ = low_rank_calculations._draw_indices(N * L + N2 * L2, self.num_components)
            nys_samples = tf.gather(
                tf.concat(
                    (tf.reshape(Xs, [N * L, -1]), tf.reshape(X2s, [N2 * L2, -1])),
                    axis=0,
                ),
                idx,
                axis=0,
            )

            Phi_lvls  = self._K_seq_lr_feat(Xs,  nys_samples=nys_samples, seeds=seeds)
            Phi2_lvls = self._K_seq_lr_feat(X2s, nys_samples=nys_samples, seeds=seeds)

            Kxx_lvls  = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi_lvls], axis=0)
            Kxx2_lvls = tf.stack([tf.matmul(Phi_lvls[i], Phi2_lvls[i], transpose_b=True)
                                  for i in range(self.num_levels + 1)], axis=0)
        else:
            Kxx_lvls  = self._K_seq(Xs)
            Kxx2_lvls = self._K_seq(Xs, X2s)

        if self.normalization:
            jit = default_jitter()
            Kxx_lvls  = Kxx_lvls  + jit * tf.eye(N,  dtype=_tf_float())[None]
            Kxx_s = tf.sqrt(tf.linalg.diag_part(Kxx_lvls))
            Kxx_lvls  = Kxx_lvls  / (Kxx_s[:, :, None] * Kxx_s[:, None, :])
            Kxx2_lvls = Kxx2_lvls / (Kxx_s[:, :, None])

        if full_X2_cov:
            if self.low_rank:
                Kx2x2_lvls = tf.stack([tf.matmul(P, P, transpose_b=True) for P in Phi2_lvls], axis=0)
            else:
                Kx2x2_lvls = self._K_seq(X2s)

            if self.normalization:
                jit = default_jitter()
                Kx2x2_lvls = Kx2x2_lvls + jit * tf.eye(N2, dtype=_tf_float())[None]
                Kx2_s = tf.sqrt(tf.linalg.diag_part(Kx2x2_lvls))
                Kxx2_lvls = Kxx2_lvls / (Kx2_s[:, None, :])
                Kx2x2_lvls = Kx2x2_lvls / (Kx2_s[:, :, None] * Kx2_s[:, None, :])

            Kxx_lvls  = Kxx_lvls  * (self.sigma * self.variances[:, None, None])
            Kxx2_lvls = Kxx2_lvls * (self.sigma * self.variances[:, None, None])
            Kx2x2_lvls = Kx2x2_lvls * (self.sigma * self.variances[:, None, None])

            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls
            else:
                return tf.reduce_sum(Kxx_lvls,  axis=0), \
                       tf.reduce_sum(Kxx2_lvls, axis=0), \
                       tf.reduce_sum(Kx2x2_lvls, axis=0)
        else:
            if self.low_rank:
                Kx2x2_lvls_diag = tf.stack([tf.reduce_sum(tf.square(P), axis=-1) for P in Phi2_lvls], axis=0)
            else:
                Kx2x2_lvls_diag = self._K_seq_diag(X2s)

            if self.normalization:
                Kx2x2_lvls_diag = Kx2x2_lvls_diag + default_jitter()
                Kx2_s = tf.sqrt(Kx2x2_lvls_diag)
                Kxx2_lvls = Kxx2_lvls / (Kx2_s[:, None, :])
                # normalized diagonal equals sigma*variances
                Kx2x2_lvls_diag = tf.tile((self.sigma * self.variances[:, None]), [1, N2])
            else:
                Kx2x2_lvls_diag = Kx2x2_lvls_diag * (self.sigma * self.variances[:, None])

            Kxx_lvls  = Kxx_lvls  * (self.sigma * self.variances[:, None, None])
            Kxx2_lvls = Kxx2_lvls * (self.sigma * self.variances[:, None, None])

            if return_levels:
                return Kxx_lvls, Kxx2_lvls, Kx2x2_lvls_diag
            else:
                return tf.reduce_sum(Kxx_lvls,  axis=0), \
                       tf.reduce_sum(Kxx2_lvls, axis=0), \
                       tf.reduce_sum(Kx2x2_lvls_diag, axis=0)

    # ------------------------------ Distance helpers (unchanged logic) ------- #
    def _square_dist(self, X, X2=None):
        """
        Works on (..., N, D) and optionally (..., M, D).
        """
        batch = tf.shape(X)[:-2]
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            dist = -2.0 * tf.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, tf.concat([batch, [-1, 1]], axis=0)) \
                 +  tf.reshape(Xs, tf.concat([batch, [1, -1]], axis=0))
            return dist
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2.0 * tf.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs,  tf.concat([batch, [-1, 1]], axis=0)) \
              + tf.reshape(X2s, tf.concat([batch, [1, -1]], axis=0))
        return dist

    def _euclid_dist(self, X, X2=None):
        r2 = self._square_dist(X, X2)
        return tf.sqrt(tf.maximum(r2, tf.cast(1e-40, _tf_float())))


# ------------------------------ Base kernel subclasses ---------------------- #
class SignatureLinear(SignatureKernel):
    """ Identity embedding """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._lin

    def _lin(self, X, X2=None):
        if X2 is None:
            return tf.matmul(X, X, transpose_b=True)
        else:
            return tf.matmul(X, X2, transpose_b=True)


class SignatureCosine(SignatureKernel):
    """ Cosine similarity embedding """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._cos

    def _cos(self, X, X2=None):
        X_norm = tf.sqrt(tf.reduce_sum(tf.square(X), axis=-1))
        if X2 is None:
            return tf.matmul(X, X, transpose_b=True) / (X_norm[..., :, None] * X_norm[..., None, :])
        else:
            X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2), axis=-1))
            return tf.matmul(X, X2, transpose_b=True) / (X_norm[..., :, None] * X2_norm[..., None, :])


class SignaturePoly(SignatureKernel):
    """ Polynomial embedding """
    def __init__(self, input_dim, num_features, num_levels, gamma=1, degree=3, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self.gamma  = Parameter(_to_float(gamma),  transform=positive())
        # In TF1 degree was non-trainable float param; keep non-trainable here:
        self.degree = Parameter(_to_float(degree), trainable=False)
        self._base_kern = self._poly

    def _poly(self, X, X2=None):
        if X2 is None:
            return (tf.matmul(X, X, transpose_b=True) + self.gamma) ** self.degree
        else:
            return (tf.matmul(X, X2, transpose_b=True) + self.gamma) ** self.degree


class SignatureRBF(SignatureKernel):
    """ Gaussian / RBF embedding """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._rbf

    def _rbf(self, X, X2=None):
        K = tf.exp(-self._square_dist(X, X2) / 2.0)
        return K

SignatureGauss = SignatureRBF


class SignatureMix(SignatureKernel):
    """ Convex combination of RBF and linear """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self.mixing = Parameter(_to_float(0.5), transform=positive())
        self._base_kern = self._mix

    def _mix(self, X, X2=None):
        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        if X2 is None:
            inner = tf.matmul(X, X, transpose_b=True)
            ds = Xs[..., :, None] + Xs[..., None, :] - 2.0 * inner
        else:
            X2s = tf.reduce_sum(tf.square(X2), axis=-1)
            inner = tf.matmul(X, X2, transpose_b=True)
            ds = Xs[..., :, None] + X2s[..., None, :] - 2.0 * inner
        K = self.mixing * tf.exp(-ds / 2.0) + (1.0 - self.mixing) * inner
        return K

class SignatureMatern32Mix(SignatureKernel):
    """Convex combination of Matérn-3/2 and Linear"""
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self.mixing = Parameter(_to_float(0.5), transform=positive())
        self._base_kern = self._mix

    def _mix(self, X, X2=None):
        # squared Euclidean distance
        if X2 is None:
            r2 = self._square_dist(X)
        else:
            r2 = self._square_dist(X, X2)

        r = tf.sqrt(tf.maximum(r2, 1e-12))   # Euclidean distance
        sqrt3 = tf.cast(np.sqrt(3.0), _tf_float())
        matern32 = (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)

        # linear part
        if X2 is None:
            linear = tf.matmul(X, X, transpose_b=True)
        else:
            linear = tf.matmul(X, X2, transpose_b=True)

        return self.mixing * matern32 + (1.0 - self.mixing) * linear
    

class SignatureMatern52Mix(SignatureKernel):
    """Convex combination of Matérn-5/2 and Linear."""
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        # If you prefer strict [0,1], swap positive() with a Sigmoid(0,1) bijector if available.
        self.mixing = Parameter(_to_float(0.5), transform=positive())
        self._base_kern = self._mix

    def _mix(self, X, X2=None):
        # Pairwise distances
        if X2 is None:
            r = self._euclid_dist(X)          # uses helper from SignatureKernel
            linear = tf.matmul(X, X, transpose_b=True)
        else:
            r = self._euclid_dist(X, X2)
            linear = tf.matmul(X, X2, transpose_b=True)

        # Matérn 5/2: (1 + sqrt(5) r + 5/3 r^2) * exp(-sqrt(5) r)
        sqrt5 = tf.cast(np.sqrt(5.0), _tf_float())
        r2 = tf.square(r)
        matern52 = (1.0 + sqrt5 * r + (5.0/3.0) * r2) * tf.exp(-sqrt5 * r)

        return self.mixing * matern52 + (1.0 - self.mixing) * linear



    # These overrides are still fine; they additionally slice tensor dims
    def _gather_feature_dims(self):
        """
        Build per-step feature indices (across lags) from active_dims, supporting slice.
        Returns tf.int32 vector or None if no-op.
        """
        ad = self.active_dims
        if ad is None:
            return None

        if isinstance(ad, slice):
            if ad.start is None and ad.stop is None and ad.step is None:
                return None
            start = 0 if ad.start is None else int(ad.start)
            stop  = self.num_features if ad.stop is None else int(ad.stop)
            step  = 1 if ad.step is None else int(ad.step)
            idx = tf.range(start, stop, delta=step, dtype=tf.int32)
        else:
            idx = tf.convert_to_tensor(ad, tf.int32)

        if self.num_lags > 0:
            offs = tf.range(self.num_lags + 1, tf.int32) * self.num_features
            idx  = tf.reshape(offs[:, None] + idx[None, :], [-1])
        return idx


    def K_tens(self, Z, *, return_levels=False, increments=False):
        idx = self._gather_feature_dims()
        Zs  = tf.gather(Z, idx, axis=-1) if idx is not None else Z
        return super().K_tens(Zs, return_levels=return_levels, increments=increments)

    def K_tens_vs_seq(self, Z, X, *, return_levels=False, increments=False, presliced=False):
        idx = self._gather_feature_dims()
        Zs  = tf.gather(Z, idx, axis=-1) if idx is not None else Z
        return super().K_tens_vs_seq(Zs, X, return_levels=return_levels,
                                     increments=increments, presliced=presliced)


class SignatureSpectral(SignatureKernel):
    """ Spectral family embedding (exp / rbf / mixed) """
    def __init__(self, input_dim, num_features, num_levels, family='gauss', Q=5, **kwargs):
        super().__init__(input_dim, num_features, num_levels, lengthscales=None, **kwargs)
        fam = str(family).lower()
        if fam in ['exp', 'exponential']: self.family = 'exp'
        elif fam in ['gauss', 'gaussian', 'rbf']: self.family = 'rbf'
        elif fam in ['mixed', 'mix']:     self.family = 'mixed'
        else: raise ValueError("Unrecognized spectral family name.")
        self.Q     = int(Q)
        self.alpha = Parameter(_to_float(np.exp(np.random.randn(self.Q))),                      transform=positive())
        self.omega = Parameter(_to_float(np.exp(np.random.randn(self.Q, self.num_features))),   transform=positive())
        self.gamma = Parameter(_to_float(np.exp(np.random.randn(self.Q, self.num_features))),   transform=positive())
        self._base_kern = self._spectral
        self.block_cols = 2048  # safe default for fp64; tune if needed

    def _spectral(self, X, X2=None):
        """
        Rank-aware spectral base: operates on the *last* dim as features.
        If X has shape (L, N, D) and X2 is None/(L, N2, D) -> returns (L, N, N2).
        Otherwise flattens leading dims (..., D) -> (N, D) and returns (N, N2).
        """
        dtype  = _tf_float()
        two_pi = tf.cast(2.0*np.pi, dtype)

        X = tf.convert_to_tensor(X, dtype=dtype)
        D = tf.shape(X)[-1]

        # --- promote inputs to (L, N, D) and (L2, N2, D) ---
        def to_rank3(T):
            r = tf.rank(T); s = tf.shape(T)
            pad = tf.maximum(0, 3 - r)
            s3 = tf.concat([tf.ones([pad], tf.int32), s], axis=0)
            return tf.reshape(T, [s3[-3], s3[-2], s3[-1]])  # (L,N,D)
        X3 = to_rank3(X)
        if X2 is None:
            X23 = X3
        else:
            X2  = tf.convert_to_tensor(X2, dtype=dtype)
            X23 = to_rank3(X2)
            tf.debugging.assert_equal(tf.shape(X3)[-1], tf.shape(X23)[-1], message="feature dims must match")

        # broadcast on level axis
        diff = X3[:, :, None, :] - X23[:, None, :, :]             # (L*, N, N2, D)

        gamma = tf.cast(self.gamma, dtype)[:, None, None, None, :]  # (Q,1,1,1,D)
        omega = tf.cast(self.omega, dtype)[:, None, None, None, :]  # (Q,1,1,1,D)
        alpha = tf.cast(self.alpha, dtype)[:, None, None, None]     # (Q,1,1,1)

        sq    = tf.reduce_sum(tf.square(diff[None, ...] * gamma), axis=-1)   # (Q,L*,N,N2)
        phase = two_pi * tf.reduce_sum(diff[None, ...] * omega, axis=-1)     # (Q,L*,N,N2)
        coss  = tf.cos(phase)

        if self.family == 'exp':
            env = tf.exp(- tf.sqrt(tf.maximum(sq, 0.0)) / 2.0)
            K = tf.reduce_sum(env * coss * alpha, axis=0)                    # (L*,N,N2)
            return K
        if self.family == 'rbf':
            env = tf.exp(- sq / 2.0)
            K = tf.reduce_sum(env * coss * alpha, axis=0)
            return K

        # 'mixed'
        Q  = tf.shape(sq)[0]
        Q1 = tf.cast(self.Q // 2, tf.int32)
        sq1, sq2 = sq[:Q1],  sq[Q1:]
        c1,  c2  = coss[:Q1], coss[Q1:]
        a1,  a2  = alpha[:Q1], alpha[Q1:]
        rbf_term = tf.exp(- sq1 / 2.0)
        exp_term = tf.exp(- tf.sqrt(tf.maximum(sq2, 0.0)) / 2.0)
        K1 = tf.reduce_sum(rbf_term * c1 * a1, axis=0)
        K2 = tf.reduce_sum(exp_term * c2 * a2, axis=0)
        return K1 + K2

    # ---- flat (N1,D) x (N2,D) blockwise pairwise kernel (no diff tensor; avoids OOM) ----
    def _spectral_blocks(self, Xf, Yf, *, block_cols=None):
        """
        Compute K(Xf,Yf) for spectral mixture WITHOUT constructing (Q×N1×N2×D).
        Xf: (N1, D), Yf: (N2, D) → returns (N1, N2).
        Processes Y in column blocks to keep peak memory small.
        """
        dtype = _tf_float()
        Xf = tf.convert_to_tensor(Xf, dtype=dtype)   # (N1,D)
        Yf = tf.convert_to_tensor(Yf, dtype=dtype)   # (N2,D)
        N1 = tf.shape(Xf)[0]
        N2 = tf.shape(Yf)[0]
        bc = tf.constant(block_cols or getattr(self, "block_cols", 2048), tf.int32)

        # loop vars are TUPLES and accumulator is 3D: (1, N1, 0)
        i0   = tf.constant(0, tf.int32)
        acc0 = tf.zeros([1, N1, 0], dtype=dtype)

        def cond(j, acc):
            return j < N2

        def body(j, acc):
            j1  = tf.minimum(j + bc, N2)
            Kbj = self._spectral(Xf, Yf[j:j1])        # (1, N1, Bj)  (rank-3 by construction)
            acc = tf.concat([acc, Kbj], axis=-1)      # (1, N1, cols+Bj)
            return j1, acc

        # IMPORTANT: shape_invariants must match the 3D loop var
        _, out3 = tf.while_loop(
            cond, body,
            loop_vars=(i0, acc0),
            shape_invariants=(i0.get_shape(), tf.TensorShape([1, None, None]))
        )
        # return 2D (N1, N2) exactly like the dense path expects
        return tf.squeeze(out3, axis=0)


class SignatureMatern12(SignatureKernel):
    """ Laplace / Exponential embedding (ν=1/2 Matérn) """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._Matern12

    def _Matern12(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        return tf.exp(-r)

SignatureLaplace     = SignatureMatern12
SignatureExponential = SignatureMatern12


class SignatureMatern32(SignatureKernel):
    """ Matérn ν=3/2 embedding """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._Matern32

    def _Matern32(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        sqrt3 = tf.cast(np.sqrt(3.0), _tf_float())
        return (1.0 + sqrt3 * r) * tf.exp(-sqrt3 * r)


class SignatureMatern52(SignatureKernel):
    """ Matérn ν=5/2 embedding """
    def __init__(self, input_dim, num_features, num_levels, **kwargs):
        super().__init__(input_dim, num_features, num_levels, **kwargs)
        self._base_kern = self._Matern52

    def _Matern52(self, X, X2=None):
        r = self._euclid_dist(X, X2)
        sqrt5 = tf.cast(np.sqrt(5.0), _tf_float())
        return (1.0 + sqrt5 * r + (5.0/3.0) * tf.square(r)) * tf.exp(-sqrt5 * r)


class SignaturePeriodic1D(SignatureKernel):
    """
    Standard periodic kernel on selected feature(s) (usually time only).
    k(x,x') = exp( - 2 * sin^2(pi (x - x') / period) / ell^2 )
    """
    def __init__(self, input_dim, num_features, num_levels,
                 period, ell=1.0, active_dims=None, **kwargs):
        super().__init__(input_dim, num_features, num_levels,
                         active_dims=active_dims, **kwargs)
        self.period = Parameter(_to_float(period), transform=positive())
        self.ell    = Parameter(_to_float(ell),    transform=positive())
        self._base_kern = self._per

    def _per(self, X, X2=None):
        """Accepts X ∈ (N,D) or (L,N,D); returns (N,N2) or (L,N,N2)."""
        dtype = _tf_float()

        def _pair(A, B):
            # A:(N,Dsel), B:(N2,Dsel)   -> (N,N2)
            a = A[..., :1]                      # time col
            b = B[..., :1]
            d = a - tf.transpose(b)             # (N,N2)
            s = tf.sin(np.pi * d / self.period)
            return tf.exp(-2.0 * tf.square(s) / (self.ell**2))

        X = tf.convert_to_tensor(X, dtype=dtype)
        if tf.rank(X) == 2:
            return _pair(X, X if X2 is None else tf.convert_to_tensor(X2, dtype))

        # rank-3 per-level: X:(L,N,Dsel), X2:(L?,N2,Dsel)
        def _to_LND(T):
            r = tf.rank(T); s = tf.shape(T)
            if r == 3: return T
            # promote 2-D to (1,N,Dsel) if ever invoked with 2-D here
            return tf.reshape(T, [1, s[-2], s[-1]])

        X3  = _to_LND(X)                         # (L,N,Dsel)
        X23 = _to_LND(X3 if X2 is None else tf.convert_to_tensor(X2, dtype))  # (L2,N2,Dsel)
        # broadcast on level axis (L*):
        diff = X3[:, :, None, : ] - X23[:, None, :, :]      # (L*,N,N2,Dsel)

        s = tf.sin(np.pi * diff / self.period)              # (L*,N,N2,Dsel)
        r2 = tf.reduce_sum(tf.square(s), axis=-1)           # (L*,N,N2)
        return tf.exp(-2.0 * r2 / (self.ell**2))

    # These two are kept to ensure proper feature slicing in tensor paths as well
    def _gather_feature_dims(self):
        ad = self.active_dims
        if ad is None:
            return None

        if isinstance(ad, slice):
            if ad.start is None and ad.stop is None and ad.step is None:
                return None
            start = 0 if ad.start is None else int(ad.start)
            stop  = self.num_features if ad.stop is None else int(ad.stop)
            step  = 1 if ad.step is None else int(ad.step)
            idx = tf.range(start, stop, delta=step, dtype=tf.int32)
        else:
            idx = tf.convert_to_tensor(ad, tf.int32)

        if self.num_lags > 0:
            offs = tf.range(self.num_lags + 1, tf.int32) * self.num_features
            idx  = tf.reshape(offs[:, None] + idx[None, :], [-1])
        return idx


    def K_tens(self, Z, *, return_levels=False, increments=False):
        idx = self._gather_feature_dims()
        Zs  = tf.gather(Z, idx, axis=-1) if idx is not None else Z
        return super().K_tens(Zs, return_levels=return_levels, increments=increments)

    def K_tens_vs_seq(self, Z, X, *, return_levels=False, increments=False, presliced=False):
        idx = self._gather_feature_dims()
        Zs  = tf.gather(Z, idx, axis=-1) if idx is not None else Z
        return super().K_tens_vs_seq(Zs, X, return_levels=return_levels,
                                     increments=increments, presliced=presliced)


class SignatureSum(SignatureKernel):
    """Sum of two SignatureKernel children (k1 + k2) with possibly different active_dims."""
    def __init__(self, k1: SignatureKernel, k2: SignatureKernel):
        assert isinstance(k1, SignatureKernel) and isinstance(k2, SignatureKernel)
        super().__init__(input_dim=k1.input_dim, num_features=k1.num_features,
                         num_levels=k1.num_levels,  # num_levels must match
                         active_dims=None,         # parent doesn’t slice; children do
                         order=k1.order,
                         normalization=k1.normalization,
                         difference=k1.difference,
                         num_lags=0, low_rank=False)
        self.k1, self.k2 = k1, k2

    # sequence kernels
    def K(self, X, X2=None, *, presliced=False, return_levels=False,
          presliced_X=False, presliced_X2=False):
        return self.k1.K(X, X2, presliced=presliced, return_levels=return_levels,
                         presliced_X=presliced_X, presliced_X2=presliced_X2) \
             + self.k2.K(X, X2, presliced=presliced, return_levels=return_levels,
                         presliced_X=presliced_X, presliced_X2=presliced_X2)

    def K_diag(self, X, *, presliced=False, return_levels=False):
        return self.k1.K_diag(X, presliced=presliced, return_levels=return_levels) \
             + self.k2.K_diag(X, presliced=presliced, return_levels=return_levels)

    # tensor paths
    def K_tens(self, Z, *, return_levels=False, increments=False):
        return self.k1.K_tens(Z, return_levels=return_levels, increments=increments) \
             + self.k2.K_tens(Z, return_levels=return_levels, increments=increments)

    def K_tens_vs_seq(self, Z, X, *, return_levels=False, increments=False, presliced=False):
        return self.k1.K_tens_vs_seq(Z, X, return_levels=return_levels,
                                     increments=increments, presliced=presliced) \
             + self.k2.K_tens_vs_seq(Z, X, return_levels=return_levels,
                                     increments=increments, presliced=presliced)


# ------------------------------ Weighted sum wrapper ------------------------ #
class SignatureWeightedSum(SignatureKernel):
    def __init__(self, k1: SignatureKernel, k2: SignatureKernel, init_rho=0.5):
        assert isinstance(k1, SignatureKernel) and isinstance(k2, SignatureKernel)
        # parent initialised from k1's config; parent sigma/variances not used
        super().__init__(input_dim=k1.input_dim,
                         num_features=k1.num_features,
                         num_levels=k1.num_levels,
                         active_dims=None,
                         order=k1.order,
                         normalization=k1.normalization,
                         difference=k1.difference,
                         num_lags=0,
                         low_rank=False)
        self.k1, self.k2 = k1, k2

        # learnable mixing rho in (0,1): use Sigmoid if available, else Positive + clipping
        if 'Sigmoid' in globals() and Sigmoid is not None:
            self.rho = Parameter(_to_float(init_rho), transform=Sigmoid(lower=0.0, upper=1.0))
        else:
            self.rho = Parameter(_to_float(init_rho), transform=positive())  # we'll clip in calls

    @property
    def _rho01(self):
        # ensure in [0,1] if using positive() fallback
        return tf.clip_by_value(tf.cast(self.rho, _tf_float()), 0.0, 1.0)

    # ---- sequence kernels ----
    def K(self, X, X2=None, *, presliced=False, return_levels=False,
          presliced_X=False, presliced_X2=False):
        r = self._rho01
        K1 = self.k1.K(X, X2, presliced=presliced, return_levels=return_levels,
                       presliced_X=presliced_X, presliced_X2=presliced_X2)
        K2 = self.k2.K(X, X2, presliced=presliced, return_levels=return_levels,
                       presliced_X=presliced_X, presliced_X2=presliced_X2)
        return r * K1 + (1.0 - r) * K2

    def K_diag(self, X, *, presliced=False, return_levels=False):
        r = self._rho01
        return r * self.k1.K_diag(X, presliced=presliced, return_levels=return_levels) \
             + (1.0 - r) * self.k2.K_diag(X, presliced=presliced, return_levels=return_levels)

    # ---- tensor paths ----
    def K_tens(self, Z, *, return_levels=False, increments=False):
        r = self._rho01
        return r * self.k1.K_tens(Z, return_levels=return_levels, increments=increments) \
             + (1.0 - r) * self.k2.K_tens(Z, return_levels=return_levels, increments=increments)

    def K_tens_vs_seq(self, Z, X, *, return_levels=False, increments=False, presliced=False):
        r = self._rho01
        return r * self.k1.K_tens_vs_seq(Z, X, return_levels=return_levels,
                                         increments=increments, presliced=presliced) \
             + (1.0 - r) * self.k2.K_tens_vs_seq(Z, X, return_levels=return_levels,
                                                 increments=increments, presliced=presliced)
