"""Dump safetensors key names for analysis."""
import sys
from pathlib import Path
from safetensors import safe_open

MODEL_PATH = Path(
    "models/StableVSR/models--claudiom4sir--StableVSR/snapshots/"
    "fddd0e3921c22a5dcc6468c56c44abe6564bacc2"
)

for subdir in ["vae", "unet", "controlnet"]:
    sf_path = MODEL_PATH / subdir
    files = list(sf_path.glob("*.safetensors"))
    for f in files:
        with safe_open(str(f), framework="numpy") as sf:
            keys = sorted(sf.keys())
        outfile = f"/tmp/{subdir}_keys.txt"
        with open(outfile, "w") as out:
            for k in keys:
                out.write(k + "\n")
        print(f"{subdir}: {len(keys)} keys -> {outfile}")
