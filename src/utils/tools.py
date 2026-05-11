import math

import torch
import numpy as np

def get_batch_label_vector(
    labels: list[str],
    label_map: dict[str, str],
) -> torch.Tensor:
    """
    Encode a batch of string labels as a multi-hot tensor.

    Column order follows ``class_list = list(label_map.values())`` (dict insertion order
    in Python 3.7+). This must match the class order used elsewhere (e.g. CLIP text
    prompts and loss indexing).

    For each entry in ``labels``, the string is split on ``'-'``. Every token that appears
    as a **key** in ``label_map`` sets the column for ``label_map[token]`` to ``1`` via
    ``class_list.index(...)``. Tokens not in ``label_map`` are ignored. A single token
    yields a one-hot row; multiple valid tokens (e.g. XD-style ``B1-0-0``) yield multi-hot.

    Args:
        labels: Batch of label strings (e.g. from the dataset CSV ``label`` column).
        label_map: Maps **token** (substring after split) to **display / prompt class name**
            stored as dict values; keys must cover every token you want to activate.

    Returns:
        Float tensor of shape ``(len(labels), len(label_map))``.
    """
    class_list = list(label_map.values())
    c = len(class_list)
    rows: list[torch.Tensor] = []

    for label in labels:
        v = torch.zeros(c)
        for token in str(label).split("-"):
            if token in label_map:
                class_name = label_map[token]
                v[class_list.index(class_name)] = 1.0
        rows.append(v.unsqueeze(0))

    return torch.cat(rows, dim=0)

def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    
    return mask.bool()

def random_extract(feat, t_max):
   r = np.random.randint(feat.shape[0] - t_max)
   return feat[r : r+t_max, :]

def uniform_extract(feat: np.ndarray, t_max: int, avg: bool = True) -> np.ndarray:
    """
    Resample a frame-level feature matrix to a fixed temporal length `t_max`.

    If ``avg`` is True (default), each output row is the mean of the frames in a
    corresponding sub-interval of ``[0, T)`` (uniform partition boundaries; adjacent
    boundaries may coincide after rounding, in which case one frame is copied).

    If ``avg`` is False, ``t_max`` frame indices are sampled uniformly in index space
    (including endpoints) and rows are taken without averaging.

    Args:
        feat: Array of shape ``[T, D]`` (``T`` frames, ``D`` per-frame dim).
        t_max: Target number of rows (output time length).
        avg: Use interval averaging if True; use discrete index sampling if False.

    Returns:
        Array of shape ``[t_max, D]``, dtype ``float32``.
    """
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    if avg:
        r = np.linspace(0, feat.shape[0], t_max + 1, dtype=np.int32)
        r = np.clip(r, 0, feat.shape[0] - 1)
        for i in range(t_max):
            if r[i]!=r[i+1]:
                new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
            else:
                new_feat[i,:] = feat[r[i],:]
    else:
        r = np.linspace(0, feat.shape[0]-1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
            
    return new_feat

def pad(feat: np.ndarray, min_len: int) -> np.ndarray:
    clip_length = feat.shape[0]
    if clip_length < min_len:
        # (a, b) means pad a elements before the row and pad b elements after the row
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    else:
       return feat

def process_feat(feat, length, is_random=False):
    """
    Pad or truncate the feature to a fixed length
    """
    clip_length = feat.shape[0]
    if feat.shape[0] > length:
        if is_random:
            return random_extract(feat, length), length
        else:
            return uniform_extract(feat, length), length
    else:
        return pad(feat, length), clip_length

def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length == 0:
        return np.empty((0, length, feat.shape[1]), dtype=np.float32), clip_length
    split_num = math.ceil(clip_length / length)
    chunks = [
        pad(feat[i * length : (i + 1) * length, :], length)
        for i in range(split_num)
    ]
    split_feat = np.stack(chunks, axis=0)
    return split_feat, clip_length