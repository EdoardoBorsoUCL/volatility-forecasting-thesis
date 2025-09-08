
from __future__ import annotations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf
import gpflow as gf


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _now() -> float:
    return time.time()

def _elbo_from_loss(loss_value: float) -> float:
    # In GPflow 2, training loss for variational models is typically -ELBO
    return -float(loss_value)

def _param_snapshot(model: gf.models.Model) -> Dict[str, np.ndarray]:
    """Snapshot all constrained parameters as {name: value} with numpy arrays."""
    snap = {}
    for p in model.trainable_parameters + model.parameters:  # de-dup below
        try:
            name = p.name
            if name in snap:
                continue
            snap[name] = np.array(p.numpy())
        except Exception:
            # Fallback to utilities (handles transformed constraints)
            pass
    # Also merge in gpflow utilities view (robust to non-trainable parameters)
    try:
        pdict = gf.utilities.parameter_dict(model)
        for par, val in pdict.items():
            name = getattr(par, "name", str(par))
            snap[name] = np.array(val.numpy())
    except Exception:
        pass
    return snap

def _varlist_snapshot(var_list: Sequence[tf.Variable]) -> Dict[str, np.ndarray]:
    return {v.name: np.array(v.numpy()) for v in var_list}

def _gather_var_groups(
    model: gf.models.Model,
    opt: Union[tf.optimizers.Optimizer, Sequence[tf.optimizers.Optimizer]],
    var_list: Optional[Union[Sequence[tf.Variable], Sequence[Sequence[tf.Variable]]]]
) -> List[Tuple[tf.optimizers.Optimizer, List[tf.Variable]]]:
    """
    Build optimizer → variable-group mapping.

    - If 'opt' is a single optimizer and var_list is None → optimize all model.trainable_variables.
    - If 'opt' is a single optimizer and var_list is a list of Variables → optimize just those.
    - If 'opt' is a list of optimizers and var_list is a list of lists:
        pairs optimizers[i] with var_list[i]; if len(opt) == len(var_list)+1, the last optimizer
        is applied to the remaining trainable variables not present in earlier groups.
    """
    if isinstance(opt, (list, tuple)):
        assert var_list is None or isinstance(var_list, (list, tuple)), \
            "When passing multiple optimizers, var_list must be a list (of lists) or None."
        var_groups: List[List[tf.Variable]] = []
        if var_list is None:
            # no explicit grouping: split evenly across optimizers
            tv = list(model.trainable_variables)
            chunks = max(1, len(tv) // max(1, len(opt)))
            var_groups = [tv[i*chunks:(i+1)*chunks] for i in range(len(opt)-1)]
            var_groups.append(tv[(len(opt)-1)*chunks:])
        else:
            # var_list may be list of lists OR list of Variables
            if len(var_list) > 0 and isinstance(var_list[0], tf.Variable):
                # treat it as one group for the first optimizer; rest get remaining
                first_group = list(var_list)  # type: ignore
                used = set(v.ref() for v in first_group)
                remaining = [v for v in model.trainable_variables if v.ref() not in used]
                if len(opt) == 1:
                    var_groups = [first_group]
                else:
                    var_groups = [first_group] + [[] for _ in range(len(opt)-2)] + [remaining]
            else:
                # list of lists
                used = set()
                var_groups = []
                for group in var_list:  # type: ignore
                    g = list(group)
                    var_groups.append(g)
                    used |= set(v.ref() for v in g)
                if len(opt) == len(var_groups) + 1:
                    remaining = [v for v in model.trainable_variables if v.ref() not in used]
                    var_groups.append(remaining)
                elif len(opt) != len(var_groups):
                    raise ValueError("len(opt) must equal len(var_list), or len(var_list)+1 for a 'remaining' group.")
        if len(var_groups) != len(opt):
            raise ValueError("Mismatch: number of optimizers and variable groups differ.")
        return [(opt[i], list(var_groups[i])) for i in range(len(opt))]
    else:
        # Single optimizer
        if var_list is None:
            return [(opt, list(model.trainable_variables))]  # type: ignore[arg-type]
        else:
            # var_list may be Variables or nested list (take union)
            if len(var_list) > 0 and isinstance(var_list[0], tf.Variable):  # type: ignore[index]
                groups = list(var_list)  # type: ignore[assignment]
            else:
                # flatten one level
                groups = [v for g in (var_list or []) for v in g]  # type: ignore[assignment]
            return [(opt, groups)]  # type: ignore[list-item]


# ---------------------------------------------------------------------
# Single-step gradient application that supports multiple optimizers.
# Computes grads once and slices per variable group.
# ---------------------------------------------------------------------
def _make_step_apply_grads():
    @tf.function  # new trace per call to optimize()
    def _step(loss_closure, opt_groups):
        # unique var list across groups
        all_vars, seen = [], set()
        for _, group in opt_groups:
            for v in group:
                if v.ref() not in seen:
                    all_vars.append(v); seen.add(v.ref())

        with tf.GradientTape() as tape:
            loss = loss_closure()
        grads = tape.gradient(loss, all_vars)
        grad_map = {v.ref(): g for v, g in zip(all_vars, grads)}

        for opt, group in opt_groups:
            pairs = [(grad_map[v.ref()], v) for v in group if grad_map[v.ref()] is not None]
            if pairs:
                opt.apply_gradients(pairs)
        return loss
    return _step



# ---------------------------------------------------------------------
# Public training entry point (ported from your GPflow-1 optimize)
# ---------------------------------------------------------------------
from typing import Union, Sequence, Optional, Callable, List, Dict, Any
import numpy as np
import tensorflow as tf
import gpflow as gf

def optimize(
    model: gf.models.Model,
    opt: Union[tf.optimizers.Optimizer, Sequence[tf.optimizers.Optimizer]],
    max_iter: int = 1000,
    print_freq: int = 1,
    save_freq: int = 50,
    val_scorer: Optional[Union[Callable[[gf.models.Model], float],
                               Sequence[Callable[[gf.models.Model], float]]]] = None,
    history: Optional[dict] = None,
    callbacks: Optional[Union[Callable[[gf.models.Model], object],
                              List[Callable[[gf.models.Model], object]]]] = None,
    save_params: bool = False,
    start_iter: int = 0,
    global_step: Optional[tf.Variable] = None,  # unused (API compat)
    var_list: Optional[Union[Sequence[tf.Variable], Sequence[Sequence[tf.Variable]]]] = None,
    save_best_params: bool = False,
    lower_is_better: bool = False,
    patience: Optional[int] = None,                 # DEPRECATED: iteration-based
    patience_saves: Optional[int] = None,           # NEW: number of save steps without improvement
    monitor_index: int = 0,                         # NEW: which val metric to monitor
    min_delta: float = 0.0,                         # NEW: require at least this improvement
    loss_closure: Optional[Callable[[], tf.Tensor]] = None,
) -> dict:
    """
    TF2/GPflow2 training with multi-metric validation & early stopping on save steps.
    - `val_scorer` can be a callable or a sequence of callables.
    - Validation values are stored as a list in rec["val"].
    - Early stopping/“best” tracking use `monitor_index` (default 0).
    - `patience_saves` counts *save steps* with no improvement (recommended).
      If only `patience` is provided, it falls back to the old iteration-based behavior.
    """
    if loss_closure is None:
        try:
            loss_closure = model.training_loss_closure()  # type: ignore
        except Exception as e:
            raise ValueError("You must provide a loss_closure()") from e

    opt_groups = _gather_var_groups(model, opt, var_list)

    # Init/continue history
    if history is None or not any(str(k).isdigit() for k in history.keys()):
        history = {}
        iter0 = 0
        start_time0 = 0.0
    else:
        numeric_keys = sorted(int(k) for k in history.keys() if str(k).isdigit())
        iter0 = numeric_keys[-1] if numeric_keys else 0
        start_time0 = float(history.get(iter0, {}).get("time", 0.0))

    if iter0 == 0 or start_iter == 0:
        print("-------------------------\n  Starting optimization  \n-------------------------")
    else:
        print("---------------------------\n  Continuing optimization  \n---------------------------")

    t0 = _now() - start_time0

    # Normalize val_scorer -> list
    if val_scorer is None:
        scorers: List[Callable[[gf.models.Model], float]] = []
    elif isinstance(val_scorer, (list, tuple)):
        scorers = list(val_scorer)
    else:
        scorers = [val_scorer]

    # Previous best
    prev_best = history.get("best", {})
    best_iter  = prev_best.get("iter", None)
    best_vals  = prev_best.get("val", None)
    if isinstance(best_vals, list):
        best_score = best_vals[monitor_index] if len(best_vals) > monitor_index else None
    else:
        best_score = best_vals  # legacy scalar
    no_improve_saves = 0

    step_fn = _make_step_apply_grads()

    for it in range(iter0 + 1, iter0 + max_iter + 1):
        # one optimizer step
        loss = step_fn(loss_closure, opt_groups)
        elbo = _elbo_from_loss(float(loss.numpy()))
        elapsed = _now() - t0

        if (it % print_freq == 0) or (it == iter0 + max_iter):
            print(f"\rIteration {it}\t|\tTime: {elapsed:.2f}\t|\tELBO: {elbo: .3f}", end="")

        if (it % save_freq == 0) or (it == iter0 + max_iter):
            rec: Dict[str, Any] = {"time": elapsed, "elbo": elbo}

            # callbacks
            if callbacks is not None:
                outs = []
                for cb in (callbacks if isinstance(callbacks, (list, tuple)) else [callbacks]):
                    try:
                        outs.append(cb(model))
                    except Exception:
                        outs.append(None)
                rec["saved"] = outs

            # validation
            vals_list = None
            if scorers:
                vals = []
                for fn in scorers:
                    try:
                        v = float(fn(model))
                    except Exception:
                        v = float("nan")
                    vals.append(v)
                vals_list = vals
                rec["val"] = vals_list
                v0 = vals_list[monitor_index] if len(vals_list) > monitor_index else float("nan")
                msg = f"{v0:.6f}" if np.isfinite(v0) else "nan"
                print(f"\t|\tVal.: {msg}", end="")

            # save params snapshot?
            if save_params:
                rec["params"] = _param_snapshot(model)

            history[it] = rec

            # early stop & best tracking on the monitored metric
            if save_best_params and vals_list is not None and len(vals_list) > monitor_index:
                score_val = vals_list[monitor_index]
                if np.isfinite(score_val):
                    improved = (best_score is None) or \
                               ((score_val < best_score - min_delta) if lower_is_better
                                else (score_val > best_score + min_delta))
                    if improved:
                        best_iter = it
                        best_score = score_val
                        no_improve_saves = 0
                        history["best"] = {
                            "iter": it,
                            "time": elapsed,
                            "elbo": elbo,
                            "val": vals_list,  # store all metrics
                            "params": (_varlist_snapshot([v for _, g in opt_groups for v in g])
                                       if var_list is not None else _param_snapshot(model))
                        }
                    else:
                        no_improve_saves += 1

                    # Preferred: save-step patience
                    if patience_saves is not None:
                        if no_improve_saves >= patience_saves:
                            print(f"\nEarly stop: no val improvement in {patience_saves} save steps.")
                            break
                    # Legacy: iteration-based patience
                    elif patience is not None and best_iter is not None:
                        if (it - best_iter) >= patience:
                            print(f"\nEarly stop: no val improvement for {patience} iterations.")
                            break

            print("")  # newline

    return history
