"""
Download a 500-sample slice of the yixuantt/MultiHopRAG dataset and save to Data/.

Usage:
  python Data/download_multihoprag.py

Notes:
- Requires network access and the `datasets` package.
- If you want a different count or output path, edit the constants below.
"""

import json
from pathlib import Path
from datasets import load_dataset

# Configuration
DATASET_NAME = "yixuantt/MultiHopRAG"
CONFIG_NAME = "MultiHopRAG"
OUTPUT_PATH = Path(__file__).resolve().parent / "multihoprag_500.json"
MAX_SAMPLES = 500


def main() -> None:
    print(f"Loading dataset {DATASET_NAME} ({CONFIG_NAME}) ...")
    ds = load_dataset(DATASET_NAME, CONFIG_NAME, split="train")
    if MAX_SAMPLES and len(ds) > MAX_SAMPLES:
        ds = ds.select(range(MAX_SAMPLES))
    records = ds.to_list()

    print(f"Saving {len(records)} samples to {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
