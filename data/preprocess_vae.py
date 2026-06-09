from huggingface_hub import snapshot_download
import os
import numpy as np
import imageio.v3 as iio
import cv2



snapshot_download(
    repo_id="madebyollin/movirec",
    repo_type="dataset",
    allow_patterns="video-patches/*",
    local_dir="./movirec"
)


# ============================================
# CONFIG
# ============================================

video_dir = "movirec/video-patches"
output_dir = "npy_files"

TARGET_SIZE = (256, 256)   # (width, height)
MAX_FRAMES = 32            # max frames per video
FRAME_STRIDE = 4           # take every 4th frame

# ============================================

os.makedirs(output_dir, exist_ok=True)

video_exts = (".mp4", ".avi", ".mov", ".mkv")

for filename in os.listdir(video_dir):

    if not filename.lower().endswith(video_exts):
        continue

    video_path = os.path.join(video_dir, filename)

    try:
        frames = []

        # Stream frames instead of loading entire video
        for idx, frame in enumerate(iio.imiter(video_path)):

            # Skip frames for temporal downsampling
            if idx % FRAME_STRIDE != 0:
                continue

            # Resize frame
            frame = cv2.resize(
                frame,
                TARGET_SIZE,
                interpolation=cv2.INTER_AREA
            )

            # Ensure uint8
            frame = frame.astype(np.uint8)

            frames.append(frame)

            # Stop early
            if len(frames) >= MAX_FRAMES:
                break

        if len(frames) == 0:
            print(f"Skipped empty video: {filename}")
            continue

        # Convert to compact numpy array
        video_array = np.stack(frames, axis=0)

        # Save
        out_name = os.path.splitext(filename)[0] + ".npy"
        out_path = os.path.join(output_dir, out_name)

        np.save(out_path, video_array)

        mb_size = video_array.nbytes / (1024 ** 2)

        print(
            f"Saved: {out_path} | "
            f"Shape: {video_array.shape} | "
            f"{mb_size:.2f} MB"
        )

    except Exception as e:
        print(f"Failed: {filename}")
        print(e)

print("Done.")