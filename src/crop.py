import cv2
import numpy as np
from pathlib import Path

import torch
from clip import clip
from PIL import Image
from tqdm import tqdm
from typing import Callable

def read_video_to_ndarray(
    video_path: str,
    stride: int = 1,
    shift: int = 0,
    convert_to_rgb: bool = False,
) -> np.ndarray:
    """
    Read a video file into a NumPy array of frames.

    This function uses OpenCV (`cv2.VideoCapture`) to decode frames and keeps
    every `stride`-th frame, optionally offset by `shift`.

    Args:
        video_path: Path to a video file (e.g., .mp4) readable by OpenCV.
        stride: Keep one frame every `stride` frames. Must be a positive integer.
        shift: Frame index offset used in the sampling rule
            `(grabbed + shift) % stride == 0`. Must be non-negative.
        convert_to_rgb: If True, convert decoded frames from OpenCV's default
            BGR color order to RGB.

    Returns:
        A `uint8` ndarray of shape `[T, H, W, 3]`, where `T` is the number of
        sampled frames. If the video contains no decodable frames (or sampling
        yields none), returns an empty array with shape `(0, 0, 0, 3)`.

    Raises:
        ValueError: If `stride <= 0` or `shift < 0`.
        FileNotFoundError: If the video cannot be opened by OpenCV.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    try:
        frames: list[np.ndarray] = []
        grabbed = 1
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (grabbed + shift) % stride == 0:
                if convert_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            grabbed += 1

        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        if grabbed % stride != 0:
            frames = frames[:-1] # drop the last frame if it's not a multiple of stride

        return np.stack(frames, axis=0).astype(np.uint8, copy=False)
    finally:
        cap.release()


def iter_videos_as_ndarray(
    video_dir: str | Path,
    recursive: bool = True,
    pattern: str = "*.mp4",
    stride: int = 1,
    shift: int = 0,
    convert_to_rgb: bool = False,
):
    """
    Iterate over videos under `video_dir` and yield (video_path, frames_ndarray).

    - frames shape: [T, H, W, 3], dtype: uint8
    - By default searches recursively for `*.mp4`.
    """
    video_dir = Path(video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"video_dir not found: {video_dir}")
    
    paths = video_dir.rglob(pattern) if recursive else video_dir.glob(pattern) # traverse all files in the directory that match the pattern
    for p in sorted(paths):
        if not p.is_file():
            continue
        frames = read_video_to_ndarray(
            str(p),
            stride=stride,
            shift=shift,
            convert_to_rgb=convert_to_rgb,
        )
        yield str(p), frames


def resize_and_crop(frames: np.ndarray, type: int, flip: bool = False):
    l = frames.shape[0]
    new_frames = []
    for i in range(l):
        frame = cv2.resize(frames[i], dsize=(340, 256))
        new_frames.append(frame)

    new_frames = np.array(new_frames)
    if type == 0: # central
        new_frames = new_frames[:, 16:240, 58:282, :]
    elif type == 1: # upper-left
        new_frames = new_frames[:, :224, :224, :]
    elif type == 2: # lower-left
        new_frames = new_frames[:, :224, -224:, :]
    elif type == 3: # upper-right
        new_frames = new_frames[:, -224:, :224, :]
    elif type == 4: # lower-right
        new_frames = new_frames[:, -224:, -224:, :]

    if flip: # horizontal flip
        for i in range(new_frames.shape[0]):
            new_frames[i] = cv2.flip(new_frames[i], 1)
    
    return new_frames


def extract_clip_features(
    frames: np.ndarray,
    model: torch.nn.Module,
    preprocess: Callable[[Image], torch.Tensor],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Extract CLIP features from a sequence of video frames using a given CLIP model and preprocess function.

    Args:
        frames (np.ndarray): Input array of video frames with shape [T, H, W, 3] and dtype uint8. Typically, these frames should be RGB images.
        model: Pretrained CLIP model for feature extraction. Must have an `.encode_image()` method.
        preprocess: Preprocessing function (such as a torchvision or CLIP transform) that converts PIL.Image to the expected model input tensor.
        device (str): Device identifier where the computation happens (e.g., 'cpu' or 'cuda').

    Returns:
        np.ndarray: Extracted video features of shape [T, d], where T is the number of frames and d is the feature dimension (depends on model).
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    video_features = []
    with torch.no_grad():
        for start in range(0, frames.shape[0], batch_size):
            batch = frames[start : start + batch_size] # NumPy object has safe clip
            images = [preprocess(Image.fromarray(frame)) for frame in batch]
            images = torch.stack(images, dim=0).to(device)
            features = model.encode_image(images)
            video_features.append(features)

    video_features = torch.cat(video_features, dim=0)
    return video_features.detach().cpu().numpy()


def save_video_clip_features(
    video_path: str,
    frames: np.ndarray,
    feature_save_dir: str | Path,
    model,
    preprocess,
    device: str,
    batch_size: int = 64,
):
    feature_save_dir = Path(feature_save_dir)
    feature_save_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem # "/home/usr/abc.mp4" -> "abc"
    for clip_id in range(5):
        for flip in (False, True):
            cropped_frames = resize_and_crop(frames, type=clip_id, flip=flip)
            video_features = extract_clip_features(
                cropped_frames,
                model,
                preprocess,
                device,
                batch_size=batch_size,
            )

            save_name = f"{video_name}__{clip_id}__{int(flip)}.npy"
            video_features = video_features.astype(np.float16, copy=False)
            np.save(feature_save_dir / save_name, video_features)

def convert_video_to_clip_features(
    video_dir: str,
    feature_save_dir: str | Path,
    pattern: str = "*.mp4",
    model_name: str = "ViT-B/16",
    stride: int = 1,
    shift: int = 0,
    batch_size: int = 64,
):
    video_dir_path = Path(video_dir)
    paths = video_dir_path.glob(pattern)
    video_paths = sorted(p for p in paths if p.is_file())

    generator = iter_videos_as_ndarray(
        video_dir=video_dir,
        stride=stride,
        shift=shift,
        convert_to_rgb=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device)

    for video_path, frames in tqdm(
        generator,
        total=len(video_paths),
        desc="Extracting CLIP features",
        unit="video",
    ):
        save_video_clip_features(
            video_path,
            frames,
            feature_save_dir,
            model,
            preprocess,
            device,
            batch_size=batch_size,
        )

if __name__ == '__main__':
    video_dir = "H:\\Datasets\\XD-Violence\\train\\videos"
    feature_save_dir = "H:\\Datasets\\XD-Violence\\train\\my-clipfeatures"
    pattern = "*.mp4"

    convert_video_to_clip_features(
        video_dir=video_dir,
        feature_save_dir=feature_save_dir,
        pattern=pattern,
        model_name="ViT-B/16",
        stride=16,
        shift=8,
        batch_size=64,
    )