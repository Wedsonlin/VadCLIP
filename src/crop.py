import cv2
import numpy as np
from pathlib import Path

import torch
from clip import clip
from PIL import Image
from tqdm import tqdm


def read_video_to_ndarray(
    video_path: str,
    stride: int = 1,
    convert_to_rgb: bool = False,
) -> np.ndarray:
    """
    Read an .mp4 (or any OpenCV-supported) video into a NumPy array.

    Returns:
        frames: uint8 ndarray with shape [T, H, W, 3].
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    try:
        frames: list[np.ndarray] = []
        grabbed = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if grabbed % stride == 0:
                if convert_to_rgb:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

            grabbed += 1

        if not frames:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)

        return np.stack(frames, axis=0).astype(np.uint8, copy=False)
    finally:
        cap.release()


def iter_videos_as_ndarray(
    video_dir: str | Path,
    recursive: bool = True,
    pattern: str = "*.mp4",
    stride: int = 1,
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
    model,
    preprocess,
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


def average_features_by_clip(features: np.ndarray, clip_len: int = 16) -> np.ndarray:
    """
    Average frame-level CLIP features over complete clips and drop the tail.

    For N input frames, the output length is floor(N / clip_len), matching the
    released VadCLIP features that align one feature with every 16 video frames.
    """
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")

    clip_count = features.shape[0] // clip_len
    if clip_count == 0:
        return np.empty((0, features.shape[1]), dtype=features.dtype)

    features = features[: clip_count * clip_len]
    return features.reshape(clip_count, clip_len, features.shape[1]).mean(axis=1)


def save_video_clip_features(
    video_path: str,
    frames: np.ndarray,
    feature_save_dir: str | Path,
    model,
    preprocess,
    device: str,
    clip_len: int = 16,
    batch_size: int = 64,
):
    feature_save_dir = Path(feature_save_dir)
    feature_save_dir.mkdir(parents=True, exist_ok=True)

    video_name = Path(video_path).stem # "/home/usr/abc.mp4" -> "abc"
    for clip_id in range(5):
        for flip in (False, True):
            cropped_frames = resize_and_crop(frames, type=clip_id, flip=flip)
            frame_features = extract_clip_features(
                cropped_frames,
                model,
                preprocess,
                device,
                batch_size=batch_size,
            )
            video_features = average_features_by_clip(frame_features, clip_len=clip_len)

            save_name = f"{video_name}__{clip_id}__{int(flip)}.npy"
            video_features = video_features.astype(np.float16, copy=False)
            np.save(feature_save_dir / save_name, video_features)


if __name__ == '__main__':
    video_dir = "H:\\Datasets\\XD-Violence\\train\\videos"
    feature_save_dir = "H:\\Datasets\\XD-Violence\\train\\my-clipfeatures"
    pattern = "*.mp4"

    video_dir_path = Path(video_dir)
    paths = video_dir_path.glob(pattern)
    video_paths = sorted(p for p in paths if p.is_file())
    generator = iter_videos_as_ndarray(
        video_dir, 
        pattern=pattern,
        stride=1, 
        convert_to_rgb=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)
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
            clip_len=16,
        )

    # video_name = "A.Beautiful.Mind.2001__#00-04-20_00-05-35_label_A"
    # video_path = "H:\\Datasets\\XD-Violence\\train\\videos\\"

    # frames = read_video_to_ndarray(
    #     video_path=video_path+video_name+".mp4",
    #     stride=1,
    #     convert_to_rgb=True,
    # )
    # cropped_frames = resize_and_crop(frames, type=0, flip=False)
    # video_features = extract_clip_features(cropped_frames, model, preprocess, device)
    # video_features = video_features.astype(np.float16)
    # video_features = average_features_by_clip(video_features)

    # np_path = "H:\\Datasets\\XD-Violence\\train\\clipfeatures\\" + video_name + "__0.npy"
    # arr = np.load(np_path)

    # print(video_features.shape)
    # print(arr.shape)

    # print(video_features[:3])
    # print(arr[:3])