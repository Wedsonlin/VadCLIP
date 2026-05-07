import numpy as np
from src.crop import read_video_to_ndarray


if __name__ == "__main__":
    video_dir = "H:\\Datasets\\XD-Violence\\test\\videos"
    feature_save_dir = "H:\\Datasets\\XD-Violence\\test\\my-clipfeatures"
    target_feature_dir = "H:\\Datasets\\XD-Violence\\test\\clipfeatures"

    import os
    for file in os.listdir(feature_save_dir):
        if file.endswith(".npy"):
            vector = np.load(os.path.join(feature_save_dir, file))
            name = file.rstrip(".npy").split("__")
            index = int(name[-1]) * 5 + int(name[-2])
            target_file = os.path.join(target_feature_dir, f"{'__'.join(name[:-2])}__{index}.npy")
            target_vector = np.load(target_file)
            if vector.shape != target_vector.shape:
                print("not match")
                print(name)
                video = read_video_to_ndarray(
                    os.path.join(video_dir, f"{name[0]}__{name[1]}.mp4"),
                    stride=1,
                    shift=0,
                    convert_to_rgb=True,
                    )
                print(file)
                print(video.shape)
                print(vector.shape)
                print(target_vector.shape)
                print(video.shape[0]%16)
            # else:
            #     print("match")
            #     video = read_video_to_ndarray(
            #         os.path.join(video_dir, f"{name[0]}__{name[1]}.mp4"),
            #         stride=1,
            #         shift=0,
            #         convert_to_rgb=True,
            #         )
            #     print(file)
            #     print(video.shape)
            #     print(vector.shape)
            #     print(video.shape[0]%16)  