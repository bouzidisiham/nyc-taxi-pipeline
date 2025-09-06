import duckdb
import pandas as pd
from pathlib import Path
import requests
import click
from src.utils_io import get_logger, ensure_dirs, read_yaml

LOG = get_logger()
NYC_TLC_BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"

def download_month(year:int, month:int, out_dir:Path) -> Path:
    fname = f"yellow_tripdata_{year}-{month:02d}.parquet"
    url = f"{NYC_TLC_BASE}/{fname}"
    out = out_dir / fname
    if not out.exists():
        LOG.info(f"Downloading {url}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        LOG.info(f"Saved to {out}")
    else:
        LOG.info(f"File exists: {out}")
    return out

@click.command()
@click.option("--year", type=int, default=2023)
@click.option("--month", type=int, default=1)
@click.option("--limit", type=int, default=None, help="optional row cap for faster dev")
def main(year:int, month:int, limit:int|None):
    cfg = read_yaml()
    raw_dir = Path(cfg["data"]["raw_dir"])
    ensure_dirs(raw_dir, Path(cfg["data"]["base_dir"]))
    path = download_month(year, month, raw_dir)
    if limit:
        LOG.info(f"Creating limited parquet with first {limit} rows")
        df = pd.read_parquet(path, engine="pyarrow")
        df = df.head(limit)
        dev_path = raw_dir / f"yellow_tripdata_{year}-{month:02d}_limited.parquet"
        df.to_parquet(dev_path, index=False)
        LOG.info(f"Wrote {dev_path}")
    LOG.info("Done.")
if __name__ == "__main__":
    main()
