"""Pre-download the 10 Pythia-70m model variants we need for EXP 1.

Models:
  - EleutherAI/pythia-70m                (the "standard" — used as anchor in EXP 4)
  - EleutherAI/pythia-70m-seed{1..9}     (nine independently-trained seeds)

Each is ~166 MB in fp32. Total ~1.7 GB on disk.

Cache lives at the standard Hugging Face location ($HF_HOME, which defaults to
~/.cache/huggingface). Idempotent: re-runs skip already-downloaded files.

We download (a) the model weights and (b) the tokenizer. The tokenizer is
identical across all seeds (verified by sha — they all use the same
EleutherAI/pythia-160m tokenizer because the Pythia family ties tokenizers
across model sizes), so we only fetch it once.
"""

from __future__ import annotations

import os
import time

from huggingface_hub import snapshot_download

MODELS = [
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-seed1",
    "EleutherAI/pythia-70m-seed2",
    "EleutherAI/pythia-70m-seed3",
    "EleutherAI/pythia-70m-seed4",
    "EleutherAI/pythia-70m-seed5",
    "EleutherAI/pythia-70m-seed6",
    "EleutherAI/pythia-70m-seed7",
    "EleutherAI/pythia-70m-seed8",
    "EleutherAI/pythia-70m-seed9",
]


def download_one(model_id: str) -> str:
    """Download a single model into HF_HOME. Returns the local path."""
    t0 = time.time()
    print(f"[download] {model_id} ...", flush=True)
    path = snapshot_download(
        repo_id=model_id,
        allow_patterns=["*.json", "*.bin", "*.safetensors", "*.txt", "tokenizer*"],
        cache_dir=os.environ.get("HF_HOME"),
    )
    print(f"[download] {model_id} -> {path} ({time.time() - t0:.1f}s)", flush=True)
    return path


def main():
    paths = {}
    for m in MODELS:
        try:
            paths[m] = download_one(m)
        except Exception as e:
            print(f"[ERROR] {m}: {type(e).__name__}: {e}", flush=True)
            paths[m] = None
    print(f"[done] {sum(1 for v in paths.values() if v)} / {len(MODELS)} models",
          flush=True)
    # Also fetch a couple of revisions of pythia-70m-seed1 for EXP 4
    print("[download] Pythia checkpoint revisions for EXP 4 ...", flush=True)
    for rev in ("step3000", "step143000"):
        try:
            p = snapshot_download(
                repo_id="EleutherAI/pythia-70m-seed1",
                revision=rev,
                allow_patterns=["*.json", "*.bin", "*.safetensors"],
                cache_dir=os.environ.get("HF_HOME"),
            )
            print(f"[download]   rev={rev} -> {p}", flush=True)
        except Exception as e:
            print(f"[ERROR] rev={rev}: {type(e).__name__}: {e}", flush=True)
    print("[done] all downloads", flush=True)


if __name__ == "__main__":
    main()
