#!/usr/bin/env python3
"""
prepare_vidgen.py — convert VidGen-1M videos + captions into the
(T,H,W,3) uint8 .npy + .txt format that train_dit.py expects.

Parallelized across CPU cores (video decode is CPU/IO bound — GPUs don't
help). Use --workers to set parallelism (default: all cores).

VidGen specifics:
  - Captions JSON is a flat list of {"vid", "caption"}; a video matches a
    caption when filename stem == vid.
  - Each clip is one continuous shot (VidGen pre-split at scene cuts), so we
    take a CONTIGUOUS frame window, not a uniform spread.
  - Clips with too many duplicate frames (decord corruption recovery) are
    rejected.

Usage:
    python prepare_vidgen.py \
        --video_dir  data/vidgen_raw/videos \
        --captions   data/vidgen_raw/VidGen_1M_video_caption.json \
        --output_dir data/videos \
        --num_frames 17 --size 128 --stride 3 --workers 16
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

# decord is imported inside the worker to avoid fork/thread issues.


def list_videos(video_dir):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return sorted(p for p in Path(video_dir).rglob("*") if p.suffix.lower() in exts)


def _read_frames(path):
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(path))
        return vr.get_batch(list(range(len(vr)))).asnumpy()
    except Exception:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path, plugin="pyav"))


def center_crop_resize(frame, size):
    h, w, _ = frame.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    sq = frame[top:top + side, left:left + side]
    return np.asarray(Image.fromarray(sq).resize((size, size), Image.BICUBIC),
                      dtype=np.uint8)


def duplicate_fraction(frames):
    if frames.shape[0] < 2:
        return 0.0
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    per_pair = diffs.reshape(diffs.shape[0], -1).mean(axis=1)
    return (per_pair < 1.0).sum() / per_pair.shape[0]


def sample_contiguous(frames, num_frames, stride, seed):
    T = frames.shape[0]
    span = (num_frames - 1) * stride + 1
    if T >= span:
        rng = np.random.RandomState(seed)
        start = rng.randint(0, T - span + 1)
        return frames[start: start + span: stride]
    take = frames[::stride][:num_frames]
    if take.shape[0] < num_frames:
        pad = np.repeat(take[-1:], num_frames - take.shape[0], axis=0)
        take = np.concatenate([take, pad], axis=0)
    return take


def worker(task):
    """
    Process one video. Returns (status, out_index, caption, npy_array_or_None).
    status ∈ {"ok","corrupt","error"}. Runs in a separate process.
    """
    idx, path_str, caption, num_frames, size, stride, max_dup_frac, seed = task
    path = Path(path_str)
    try:
        frames = _read_frames(path)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            return ("error", idx, None, None)
        if duplicate_fraction(frames) > max_dup_frac:
            return ("corrupt", idx, None, None)
        frames = sample_contiguous(frames, num_frames, stride, seed)
        out = np.stack([center_crop_resize(f, size) for f in frames], axis=0)
        if out.shape != (num_frames, size, size, 3) or out.dtype != np.uint8:
            return ("error", idx, None, None)
        return ("ok", idx, caption, out)
    except Exception:
        return ("error", idx, None, None)


def load_captions(path):
    data = json.loads(Path(path).read_text())
    return {o["vid"]: o["caption"].strip()
            for o in data if o.get("vid") and o.get("caption")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_dir",  required=True)
    ap.add_argument("--captions",   required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_frames", type=int, default=17)
    ap.add_argument("--size",       type=int, default=128)
    ap.add_argument("--stride",     type=int, default=1)
    ap.add_argument("--max_dup_frac", type=float, default=0.3)
    ap.add_argument("--workers",    type=int, default=os.cpu_count())
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    captions = load_captions(args.captions)
    videos = list_videos(args.video_dir)
    print(f"Loaded {len(captions)} captions, {len(videos)} videos. "
          f"Workers: {args.workers}")

    # Build task list: only videos with a caption match. Assign a stable
    # output index by enumeration order so parallel writes never collide.
    tasks = []
    no_cap = 0
    out_index = 0
    for path in videos:
        cap = captions.get(path.stem)
        if cap is None:
            no_cap += 1
            continue
        tasks.append((out_index, str(path), cap, args.num_frames, args.size,
                      args.stride, args.max_dup_frac, args.seed + out_index))
        out_index += 1

    print(f"{len(tasks)} videos to process ({no_cap} had no caption).")

    written, corrupt, errors = 0, 0, 0
    out_dir = Path(args.output_dir)

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for done, fut in enumerate(as_completed(futures), 1):
            status, idx, cap, arr = fut.result()
            if status == "ok":
                base = f"clip_{idx:06d}"
                np.save(out_dir / f"{base}.npy", arr)
                (out_dir / f"{base}.txt").write_text(cap, encoding="utf-8")
                written += 1
            elif status == "corrupt":
                corrupt += 1
            else:
                errors += 1

            if done % 200 == 0:
                print(f"  processed {done}/{len(tasks)} "
                      f"(written {written}, corrupt {corrupt}, err {errors})")

    print(f"\nDone. Wrote {written} clips. "
          f"Corrupt {corrupt}, errors {errors}, no-caption {no_cap}.")

    # Sanity check one output
    sample_files = sorted(out_dir.glob("clip_*.npy"))
    if sample_files:
        s = np.load(sample_files[0])
        print(f"Sanity ({sample_files[0].name}): dtype={s.dtype}, "
              f"shape={s.shape}, min={s.min()}, max={s.max()}")
        assert s.dtype == np.uint8


if __name__ == "__main__":
    main()