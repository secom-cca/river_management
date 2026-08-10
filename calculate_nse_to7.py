"""Calculate NSE for the present/base upstream and downstream flow series."""

import csv
import math
from pathlib import Path


RUN_DATE = "260723_1430"
DATA_DIR = Path("data")
EXTRACTED_CSV = DATA_DIR / f"para_to7_{RUN_DATE}_present_base_nse_flows.csv"
DAILY_CSV = DATA_DIR / f"para_to7_all_scenarios_{RUN_DATE}_daily.csv"
OUTPUT_CSV = DATA_DIR / f"para_to7_{RUN_DATE}_present_base_nse_summary.csv"
# Initial storage is set from an approximately 90-day residence-time assumption.
WARMUP_DAYS = 90


class NseAccumulator:
    def __init__(self):
        self.count = 0
        self.observed_sum = 0.0
        self.observed_sum_sq = 0.0
        self.squared_error_sum = 0.0
        self.skipped = 0

    def add(self, observed_text, simulated_text):
        try:
            observed = float(observed_text)
            simulated = float(simulated_text)
        except (TypeError, ValueError):
            self.skipped += 1
            return

        if not math.isfinite(observed) or not math.isfinite(simulated):
            self.skipped += 1
            return

        self.count += 1
        self.observed_sum += observed
        self.observed_sum_sq += observed * observed
        self.squared_error_sum += (simulated - observed) ** 2

    def summary(self, label):
        if self.count < 2:
            raise ValueError(f"Insufficient valid data for {label} NSE.")

        observed_mean = self.observed_sum / self.count
        observed_variance_sum = self.observed_sum_sq - self.count * observed_mean**2
        if observed_variance_sum <= 0:
            raise ValueError(f"Observed flow has no variance for {label} NSE.")

        return {
            "location": label,
            "warmup_days_excluded": WARMUP_DAYS,
            "nse": 1 - self.squared_error_sum / observed_variance_sum,
            "rmse": math.sqrt(self.squared_error_sum / self.count),
            "valid_days": self.count,
            "skipped_days": self.skipped,
            "observed_mean": observed_mean,
            "squared_error_sum": self.squared_error_sum,
            "observed_variance_sum": observed_variance_sum,
        }


def input_csv():
    if EXTRACTED_CSV.exists():
        return EXTRACTED_CSV, False
    if DAILY_CSV.exists():
        return DAILY_CSV, True
    raise FileNotFoundError(f"Neither input CSV was found: {EXTRACTED_CSV} or {DAILY_CSV}")


def main():
    source_csv, needs_filter = input_csv()
    upstream = NseAccumulator()
    downstream = NseAccumulator()

    with source_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        for row in reader:
            if needs_filter and (row["scenario"] != "present" or row["case_id"] != "base"):
                continue
            if float(row["time"]) < WARMUP_DAYS:
                continue
            upstream.add(row["flow"], row["river_discharge_upstream"])
            downstream.add(row["flow_d"], row["river_discharge_downstream"])

    summaries = [upstream.summary("upstream"), downstream.summary("downstream")]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    for summary in summaries:
        print(
            f"{summary['location']}: NSE={summary['nse']:.4f}, "
            f"RMSE={summary['rmse']:.2f}, valid days={summary['valid_days']}, "
            f"warm-up={WARMUP_DAYS} days"
        )
    print(f"Saved NSE summary to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
