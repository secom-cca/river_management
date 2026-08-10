"""Combine compatible completed to7 optimisation runs by climate scenario."""

import csv
from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_STAMP = "260728_merged"
# Each source must contain a disjoint scenario set and identical optimisation
# decision/output columns. Retain provenance in the merged CSVs.
SOURCE_STAMPS = ["260725_2203", "260726_1459"]
KINDS = ["summary", "decisions", "yearly"]


def combine_kind(kind):
    combined_rows = []
    expected_columns = None
    seen_keys = set()

    for stamp in SOURCE_STAMPS:
        source = DATA_DIR / f"opt_to7_standard_{stamp}_{kind}.csv"
        if not source.exists():
            raise FileNotFoundError(f"Missing source file: {source}")
        with source.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames
            if expected_columns is None:
                expected_columns = columns
            elif columns != expected_columns:
                raise ValueError(f"Column mismatch in {source}")

            for row in reader:
                key = (row["scenario"], row["pattern"], row.get("seed", ""))
                if kind == "yearly":
                    key += (row["year"],)
                if key in seen_keys:
                    raise ValueError(f"Duplicate scenario/pattern row in {source}: {key}")
                seen_keys.add(key)
                row["source_run_stamp"] = stamp
                combined_rows.append(row)

    output = DATA_DIR / f"opt_to7_standard_{OUTPUT_STAMP}_{kind}.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=[*expected_columns, "source_run_stamp"])
        writer.writeheader()
        writer.writerows(combined_rows)
    return output, len(combined_rows)


def main():
    outputs = [combine_kind(kind) for kind in KINDS]
    for output, count in outputs:
        print(f"Saved {count} rows: {output}")


if __name__ == "__main__":
    main()
