"""Helper script to unpack the full twcs.csv dataset from split archives."""
import io
import zipfile
from pathlib import Path

TWCS_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = TWCS_DIR / "twcs.csv"

def unpack() -> None:
    if OUTPUT_CSV.exists():
        print(f"Dataset already exists at {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size:,} bytes).")
        return

    part_files = sorted(TWCS_DIR.glob("twcs.zip.*"))
    if not part_files:
        print(f"No archive parts found in {TWCS_DIR}.")
        return

    print(f"Rebuilding twcs archive from {len(part_files)} parts...")
    combined = bytearray()
    for p in part_files:
        combined.extend(p.read_bytes())

    print(f"Extracting twcs.csv ({len(combined):,} compressed bytes)...")
    with zipfile.ZipFile(io.BytesIO(combined)) as z:
        z.extract("twcs.csv", TWCS_DIR)

    print(f"Successfully unpacked twcs.csv to {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size:,} bytes)!")

if __name__ == "__main__":
    unpack()
