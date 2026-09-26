"""Step 1 of the pipeline: normalize every source file -> data/clean/<split>_source<k>.parquet

usage: python src/preprocess.py [--data-dir DIR] [--out-dir DIR] [--splits train test]
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from normalize import normalize_record

DEFAULT_DATA = "C:/Users/malla/Downloads/6ab10eb3b23ba_student_resource/student_resource/dataset"
DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "clean")


def read_tsv(path):
    # quoting=3: names contain stray quotes ('"""') that must not be parsed as CSV quoting
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def _work(chunk):
    return [normalize_record(n, a, c) for n, a, c in chunk]


def normalize_df(df, procs):
    rows = list(zip(df.business_name, df.business_address, df.country))
    step = 20000
    chunks = [rows[i:i + step] for i in range(0, len(rows), step)]
    with Pool(procs) as pool:
        out = [r for part in pool.imap(_work, chunks) for r in part]
    norm = pd.DataFrame(out, index=df.index)
    return pd.concat([df, norm], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits:
        for k in (1, 2, 3):
            src = os.path.join(args.data_dir, split, f"{split}_source{k}.tsv")
            dst = os.path.join(args.out_dir, f"{split}_source{k}.parquet")
            t = time.time()
            df = read_tsv(src)
            df = normalize_df(df, args.procs)
            df.to_parquet(dst, index=False)
            print(f"{split} source{k}: {len(df):,} rows -> {dst}  ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
