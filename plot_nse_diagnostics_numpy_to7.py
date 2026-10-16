"""Create the existing NSE diagnostic figures from the NumPy result."""

from pathlib import Path

import plot_nse_diagnostics_to7 as diagnostics


diagnostics.DATA_DIR = Path("data/nse_to7_numpy")
diagnostics.FIG_DIR = Path("figures/nse_to7_numpy")


if __name__ == "__main__":
    diagnostics.main()
