from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any


INPUT_PATH = Path(r"/mnt/f/school/P29-21_GML/P29-21.geojson")
OUTPUT_PATH = Path("data/P29-21.csv")

# Keep the output compact and stable for downstream calculations.
FIELDS = [
    "P29_001",
    "P29_002",
    "P29_003",
    "P29_004",
    "P29_005",
    "P29_006",
    "P29_007",
    "lon",
    "lat",
]


def _extract_lon_lat(geometry: dict[str, Any]) -> tuple[Any, Any]:
    coords = geometry.get("coordinates") or []
    if geometry.get("type") == "Point" and len(coords) >= 2:
        return coords[0], coords[1]
    return "", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert school GeoJSON to CSV.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for feature in features:
            props = feature.get("properties", {}) or {}
            lon, lat = _extract_lon_lat(feature.get("geometry", {}) or {})
            row = {key: props.get(key, "") for key in FIELDS}
            row["lon"] = lon
            row["lat"] = lat
            writer.writerow(row)

    print(f"Wrote {len(features)} rows to {args.output}")


if __name__ == "__main__":
    main()
