#!/usr/bin/env python3
"""
download_vidgen.py — fetch a size-capped subset of VidGen-1M.

VidGen-1M (Fudan-FUXI/VIDGEN-1M on HF) is 2.24 TB split into 2048 shards,
plus a caption JSON keyed by 'vid'. This script:
  1. downloads the caption JSON (small),
  2. lists the shard files via the HF API,
  3. downloads shards one at a time until a size budget is reached.

So with --max_gb 28 you get ~25 shards (~1.1 GB each) instead of all 2 TB.

Requires: pip install huggingface_hub
Usage:
    python download_vidgen.py --out_dir data/vidgen_raw --max_gb 28
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "Fudan-FUXI/VIDGEN-1M"
REPO_TYPE = "dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/vidgen_raw")
    ap.add_argument("--max_gb", type=float, default=28.0,
                    help="Stop after roughly this many GB of video shards")
    ap.add_argument("--shard_ext", default=".tar",
                    help="Shard file extension to match (.tar/.zip/.tar.gz). "
                         "Auto-detected if left default and no .tar found.")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    api = HfApi()

    # List every file in the repo
    files = api.list_repo_files(REPO, repo_type=REPO_TYPE)

    # 1. Caption / metadata JSON(s) — grab all .json (they're small)
    jsons = [f for f in files if f.lower().endswith(".json")]
    print(f"Found {len(jsons)} json file(s): {jsons}")
    for j in jsons:
        print(f"  downloading {j} ...")
        hf_hub_download(REPO, j, repo_type=REPO_TYPE,
                        local_dir=str(out))

    # 2. Identify shard files. Try the requested ext; auto-detect if none.
    def shards_with(ext):
        return sorted(f for f in files if f.lower().endswith(ext))

    shards = shards_with(args.shard_ext)
    if not shards:
        # Auto-detect among common archive/video container extensions
        for ext in (".tar", ".zip", ".tar.gz", ".webdataset", ".parquet"):
            shards = shards_with(ext)
            if shards:
                print(f"[auto] using shard extension '{ext}'")
                break
    if not shards:
        non_json = [f for f in files if not f.lower().endswith(".json")]
        print("No standard shard extension matched. Non-JSON files look like:")
        for f in non_json[:20]:
            print("   ", f)
        print("Re-run with --shard_ext set to the right extension.")
        return

    print(f"Found {len(shards)} shards. Budget: {args.max_gb} GB.")

    # 3. Download shards until budget hit
    budget_bytes = args.max_gb * (1024 ** 3)
    total = 0
    got = 0
    for s in shards:
        if total >= budget_bytes:
            break
        try:
            path = hf_hub_download(REPO, s, repo_type=REPO_TYPE,
                                   local_dir=str(out))
            sz = Path(path).stat().st_size
            total += sz
            got += 1
            print(f"  [{got}] {s}  ({sz/1e9:.2f} GB)  "
                  f"running total {total/1e9:.2f} GB")
        except Exception as e:
            print(f"  [skip] {s}: {e}")

    print(f"\nDone. {got} shards, {total/1e9:.2f} GB in {out}")
    print("Next: extract the shards, then run prepare_dataset.py "
          "(use the caption JSON, keyed by 'vid').")


if __name__ == "__main__":
    main()
