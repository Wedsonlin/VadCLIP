import math

import torch
import numpy as np

def get_batch_label(texts, prompt_text, label_map: dict):
    label_vectors = torch.zeros(0)
    if len(label_map) != 7:
        if len(label_map) == 2:
            for text in texts:
                label_vector = torch.zeros(2)
                if text == 'Normal':
                    label_vector[0] = 1
                else:
                    label_vector[1] = 1
                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
        else:
            for text in texts:
                label_vector = torch.zeros(len(prompt_text))
                if text in label_map:
                    label_text = label_map[text]
                    label_vector[prompt_text.index(label_text)] = 1

                label_vector = label_vector.unsqueeze(0)
                label_vectors = torch.cat([label_vectors, label_vector], dim=0)
    else:
        for text in texts:
            label_vector = torch.zeros(len(prompt_text))
            labels = text.split('-')
            for label in labels:
                if label in label_map:
                    label_text = label_map[label]
                    label_vector[prompt_text.index(label_text)] = 1
            
            label_vector = label_vector.unsqueeze(0)
            label_vectors = torch.cat([label_vectors, label_vector], dim=0)

    return label_vectors

def get_prompt_text(label_map: dict):
    prompt_text = []
    for v in label_map.values():
        prompt_text.append(v)

    return prompt_text

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