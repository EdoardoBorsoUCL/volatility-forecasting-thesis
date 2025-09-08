import tensorflow as tf
import gpflow as gf
from gpflow.config import default_jitter
from gpflow.covariances.dispatch import Kuu, Kuf
from gpflow.conditionals.util import base_conditional

import inducing_variables as gind 

class SVGPForSignatures(gf.models.SVGP):
    @tf.function(experimental_relax_shapes=True)
    def predict_f(self, Xnew, full_cov: bool = False, full_output_cov: bool = False):
        if full_output_cov:
            raise NotImplementedError("full_output_cov=True not supported for signature path.")

        iv = self.inducing_variable
        Xnew = tf.convert_to_tensor(Xnew, dtype=gf.config.default_float())
        tf.debugging.assert_rank_at_least(Xnew, 3, message="Xnew must be (N,L,d)")

        if isinstance(iv, (gind.InducingTensors, gind.InducingSequences)):
            if full_cov:
                raise NotImplementedError("full_cov=True not implemented (no Kff).")

            # --- Kuu/Kuf with correct order & shapes ---
            Kmm = Kuu(iv, self.kernel, jitter=default_jitter())       # [M, M]
            Kmn = Kuf(iv, self.kernel, Xnew)                           # should be [M, N]
            Kmn = tf.convert_to_tensor(Kmn)
            tf.debugging.assert_rank(Kmn, 2, message="Kuf must return (M,N)")
            tf.debugging.assert_equal(tf.shape(Kmm)[0], tf.shape(Kmn)[0],
                                      message="Kuu/Kuf M mismatch")

            # --- Knn: O(N) vector, never NxN ---
            if getattr(self.kernel, "normalization", False):
                # amplitude per level → scalar sum
                amp = self.kernel.sigma * self.kernel.variances        # (L+1,)
                amp_sum = tf.reduce_sum(tf.cast(amp, gf.config.default_float()))  # scalar
                Knn = tf.fill([tf.shape(Xnew)[0]], amp_sum)            # (N,)
            else:
                # Provide a fast O(N) diag in your kernel (no NxN inside)
                Knn = self.kernel.K_diag_fast(Xnew)                    # (N,)

            # --- conditional ---
            Fmu, Fvar = base_conditional(
                Kmn=Kmn, Knn=Knn, Kmm=Kmm,
                f=self.q_mu, full_cov=False,
                q_sqrt=self.q_sqrt, white=self.whiten,
            )

            # mean_function: ensure matching shape
            if self.mean_function is not None:
                mf = self.mean_function(Xnew)                          # expect (N, P)
                tf.debugging.assert_shapes([
                    (Fmu, ("N", "P")),
                    (mf,  ("N", "P")),
                ], message="mean_function output must match Fmu")
                Fmu = Fmu + mf

            return Fmu, Fvar

        # Fallback to standard GPflow for non-custom inducing types
        return super().predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
