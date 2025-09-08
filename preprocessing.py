
from __future__ import annotations
from functools import partial
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


def pad_sequence(max_length: int, pre: bool, seq: np.ndarray) -> np.ndarray:
    """
    Pad a (len_examples, num_features) sequence to max_length by repeating
    the first (if pre=True) or last (if pre=False) row.

    Parameters
    ----------
    max_length : int
        Target length (must be >= seq.shape[0]).
    pre : bool
        True  -> pad at the beginning with the first row.
        False -> pad at the end with the last row.
    seq : np.ndarray
        Shape (len_examples, num_features).

    Returns
    -------
    np.ndarray
        Shape (max_length, num_features).
    """
    if seq.ndim != 2:
        raise ValueError("pad_sequence expects seq with shape (len_examples, num_features).")
    if max_length < seq.shape[0]:
        raise ValueError("max_length must be >= current sequence length.")

    pad_rows = max_length - seq.shape[0]
    if pad_rows == 0:
        return seq

    if pre:
        pad_block = np.tile(seq[0], (pad_rows, 1))
        return np.concatenate((pad_block, seq), axis=0)
    else:
        pad_block = np.tile(seq[-1], (pad_rows, 1))
        return np.concatenate((seq, pad_block), axis=0)


def tabulate_list_of_sequences(
    sequences_list: List[np.ndarray],
    orient_ax: int = 0,
    pad_with: Optional[float] = None,
    pre: bool = False,
) -> np.ndarray:
    """
    Convert a list of sequences to a 3D array (N, max_len, num_features), padding as needed.

    Notes
    -----
    - If sequences have different lengths, they’re padded with the first/last value (or a
      constant 'pad_with' if provided).
    - If 'orient_ax' == 1, each sequence is transposed first (num_features, len_examples) -> (len_examples, num_features).

    Parameters
    ----------
    sequences_list : list of np.ndarray
        Each with shape (len_examples, num_features) or (num_features, len_examples) if orient_ax == 1.
    orient_ax : int, default 0
        0 -> sequences are (len_examples, num_features).
        1 -> sequences are (num_features, len_examples) and will be transposed.
    pad_with : Optional[float], default None
        If None, pad by repeating edge values. Otherwise pad with this constant.
    pre : bool, default False
        If True, pad at the beginning; else pad at the end.

    Returns
    -------
    np.ndarray
        Shape (N, max_len, num_features).
    """
    if not all(seq.ndim == 2 for seq in sequences_list):
        raise ValueError("Make sure ndim == 2 for all sequences in the list!")

    # Re-orient if needed
    if orient_ax == 1:
        sequences_list = [seq.T for seq in sequences_list]

    feature_ax = 1
    num_features_each = np.asarray([seq.shape[feature_ax] for seq in sequences_list])
    if not np.all(num_features_each == num_features_each[0]):
        raise ValueError(
            "Different path dimensions found. Ensure all sequences have the same number of features."
        )
    max_length = int(max(seq.shape[0] for seq in sequences_list))

    if pad_with is None:
        pad_fn = partial(pad_sequence, max_length, pre)
    else:
        if pre:
            pad_fn = lambda x: np.concatenate(
                (np.full((max_length - x.shape[0], x.shape[1]), float(pad_with)), x),
                axis=0,
            )
        else:
            pad_fn = lambda x: np.concatenate(
                (x, np.full((max_length - x.shape[0], x.shape[1]), float(pad_with))),
                axis=0,
            )

    sequences_list_tabulated = list(tqdm(map(pad_fn, sequences_list), total=len(sequences_list)))
    sequences_array = np.stack(sequences_list_tabulated, axis=0)  # (N, max_len, num_features)
    return sequences_array


def add_time_to_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Given a (len_examples, num_features) sequence (possibly padded with repeated last rows),
    add a time coordinate in [0,1] that stops increasing once padded region begins,
    then flatten to shape (len_examples * (num_features + 1),).

    Parameters
    ----------
    sequence : np.ndarray
        Shape (len_examples, num_features).

    Returns
    -------
    np.ndarray
        Flattened shape (len_examples * (num_features + 1),).
    """
    if sequence.ndim != 2:
        raise ValueError("add_time_to_sequence expects (len_examples, num_features).")

    length, num_features = sequence.shape
    # detect trailing duplicates (padding)
    num_repeating = 1
    while num_repeating < length and np.array_equal(sequence[-1 - num_repeating], sequence[-1]):
        num_repeating += 1
    num_repeating -= 1
    unique_length = length - num_repeating

    # time in [0,1], constant over the padded tail
    time = np.arange(unique_length, dtype=np.float64) / max(unique_length - 1, 1)
    time = np.concatenate((time, np.tile(time[-1], (num_repeating,))), axis=0) if num_repeating > 0 else time

    seq_with_t = np.concatenate((time[:, None], sequence), axis=1)
    return seq_with_t.flatten()


def add_time_to_table(sequences_array: np.ndarray, num_features: Optional[int] = None) -> np.ndarray:
    """
    Add time coordinate to each sequence in a batched table and flatten.

    Parameters
    ----------
    sequences_array : np.ndarray
        Either (N, max_len, num_features) or (N, max_len * num_features).
    num_features : Optional[int]
        If sequences_array is flat (2D), provide num_features (defaults to 1).

    Returns
    -------
    np.ndarray
        Shape (N, (num_features + 1) * max_len).
    """
    if sequences_array.ndim == 3:
        if num_features is None:
            num_features = sequences_array.shape[2]
        else:
            assert num_features == sequences_array.shape[2]
    else:
        num_features = num_features or 1

    N = sequences_array.shape[0]
    sequences_array = sequences_array.reshape(N, -1, num_features)
    sequences_with_time = np.apply_along_axis(add_time_to_sequence, 1, sequences_array)
    return sequences_with_time


def add_natural_parametrization_to_table(sequences_array: np.ndarray, num_features: Optional[int] = None) -> np.ndarray:
    """
    Add natural parametrization (cumulative arc-length) as extra coordinate.

    Parameters
    ----------
    sequences_array : np.ndarray
        Either (N, max_len, num_features) or (N, max_len * num_features).
    num_features : Optional[int]
        If sequences_array is flat (2D), provide num_features (defaults to 1).

    Returns
    -------
    np.ndarray
        Shape (N, (num_features + 1) * max_len) in a 3D then flattened manner.
    """
    if sequences_array.ndim == 3:
        if num_features is None:
            num_features = sequences_array.shape[2]
        else:
            assert num_features == sequences_array.shape[2]
    else:
        num_features = num_features or 1

    N = sequences_array.shape[0]
    seqs = sequences_array.reshape(N, -1, num_features)

    # per-sequence incremental Euclidean distances and cumulative sum
    nat = np.linalg.norm(np.diff(seqs, axis=1), axis=2)  # (N, max_len-1)
    nat = np.concatenate((np.zeros((N, 1), dtype=np.float64), nat), axis=1)  # prepend 0
    nat = np.cumsum(nat, axis=1)  # (N, max_len)

    seqs_with_np = np.concatenate((nat[:, :, None], seqs), axis=2)  # (N, max_len, 1+num_features)
    return seqs_with_np


def add_time_to_list(sequences_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    For a list of (len_i, d) arrays, prepend a time coordinate t in (0,1].

    Parameters
    ----------
    sequences_list : list of np.ndarray
        Each with shape (len_i, d).

    Returns
    -------
    list of np.ndarray
        Each with shape (len_i, d+1), where the first column is t.
    """
    out = []
    for x in sequences_list:
        T = x.shape[0]
        t = (np.arange(1, T + 1, dtype=np.float64) / float(T)).reshape(T, 1)
        out.append(np.concatenate((t, x), axis=1))
    return out


def add_natural_parametrization_to_list(sequences_list: List[np.ndarray]) -> List[np.ndarray]:
    """
    For a list of (len_i, d) arrays, prepend natural parametrization (cumulative length).

    Parameters
    ----------
    sequences_list : list of np.ndarray
        Each with shape (len_i, d).

    Returns
    -------
    list of np.ndarray
        Each with shape (len_i, d+1), first column is cumulative arc-length.
    """
    out = []
    for x in sequences_list:
        diffs = np.linalg.norm(np.diff(x, axis=0), axis=1)  # (len_i - 1,)
        arc = np.cumsum(np.concatenate(([0.0], diffs), axis=0))[:, None]  # (len_i, 1)
        out.append(np.concatenate((arc, x), axis=1))
    return out
