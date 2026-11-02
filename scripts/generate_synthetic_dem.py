import argparse
from pathlib import Path
import numpy as np

from src.terrain_map import make_synthetic_dem
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/synthetic/dem.npy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dem = make_synthetic_dem(
        rows=cfg["dem"]["rows"],
        cols=cfg["dem"]["cols"],
        seed=cfg["seed"],
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, dem)
    print(f"Saved synthetic DEM {dem.shape} -> {out}")


if __name__ == "__main__":
    main()
