#!/usr/bin/env python3
"""
from WOMD→HPTR pipeline generate randomly select 2 scenes from *.npz, 
store into data/ for dry‑run use
"""

"""
run: 
python scripts/make_sample_scene.py
"""

import numpy as np
import random
import pathlib
import shutil

SRC_DIR = pathlib.Path("data/womd_hptr")     # transfered .npz folder
DST_DIR = pathlib.Path("data")
DST_DIR.mkdir(exist_ok=True, parents=True)

all_files = sorted(SRC_DIR.glob("*.npz"))
sample_files = random.sample(all_files, 2)

for src in sample_files:
    dst = DST_DIR / f"sample_{src.name}"
    shutil.copy(src, dst)
    print(f"Copied {src.name} -> {dst.name}")

# generate sample_list.txt
with open(DST_DIR / "sample_list.txt", "w") as f:
    for src in sample_files:
        f.write(f"{DST_DIR / ('sample_' + src.name)}\n")
print("✅  sample_list.txt generated!")
