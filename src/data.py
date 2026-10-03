"""Loading the Protein Tertiary Structure dataset (UCI id 265).

Phase 1 adds splitting and preprocessing to this module.

Note: the `ucimlrepo` snippet in the spec does NOT work for this dataset. UCI reports id 265 as
"not available for import". So, as the spec allows, we download the CSV from the dataset page's
direct link instead.
"""

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DOWNLOAD_URL = (
    "https://archive.ics.uci.edu/static/public/265/"
    "physicochemical+properties+of+protein+tertiary+structure.zip"
)
CSV_NAME_IN_ZIP = "CASP.csv"

TARGET = "RMSD"
FEATURES = [f"F{i}" for i in range(1, 10)]  # F1 ... F9

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "data" / "raw" / "protein.csv"


def load_raw(cache_path: Path = RAW_CSV) -> pd.DataFrame:
    """Return the full dataset as one DataFrame: columns F1..F9, then RMSD.

    The first call downloads the zip from UCI and saves the CSV. Later calls read the cached CSV,
    so they work offline and always see exactly the same data.
    """
    if cache_path.exists():
        return pd.read_csv(cache_path)

    with urllib.request.urlopen(DOWNLOAD_URL) as response:
        zip_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        df = pd.read_csv(archive.open(CSV_NAME_IN_ZIP))

    df = df[FEATURES + [TARGET]]  # the CSV has RMSD first. We put features first, target last

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df
