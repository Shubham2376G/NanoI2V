import os
import numpy as np
from pathlib import Path

os.makedirs("data/smoke", exist_ok=True)

T, H, W = 17, 128, 128

for i in range(10):
    video = np.random.randint(0, 256, (T, H, W, 3), dtype=np.uint8)
    np.save(f"data/smoke/{i:04d}.npy", video)

    with open(f"data/smoke/{i:04d}.txt", "w") as f:
        f.write("a simple test video")