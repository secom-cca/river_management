from __future__ import annotations

import json
from pathlib import Path
from typing import Any


try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "h5py is required. Install it with: python -m pip install h5py"
    ) from exc


DEFAULT_PATH = Path(r"D:\Data\NIES2020\predict\allgrid_monthly\ensemble\WBGT_ensemble_ssp245_2030-2059.h5")


def _format_attrs(attrs: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        out[str(key)] = value
    return out


def main() -> None:
    path = DEFAULT_PATH
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    print(f"file: {path}")
    print(f"size_bytes: {path.stat().st_size}")

    with h5py.File(path, "r") as f:
        print("root_attrs:", json.dumps(_format_attrs(f.attrs), ensure_ascii=False))
        print("root_keys:", list(f.keys()))

        def visitor(name: str, obj: Any) -> None:
            kind = "group" if isinstance(obj, h5py.Group) else "dataset"
            print(f"{kind}: {name}")
            print("  attrs:", json.dumps(_format_attrs(obj.attrs), ensure_ascii=False))
            if isinstance(obj, h5py.Dataset):
                print("  shape:", obj.shape)
                print("  dtype:", str(obj.dtype))
                print("  chunks:", obj.chunks)
                print("  compression:", obj.compression)
                print("  maxshape:", obj.maxshape)
                if obj.ndim > 0 and obj.size > 0:
                    sample = obj[0]
                    if hasattr(sample, "tolist"):
                        sample = sample.tolist()
                    print("  sample0:", sample)

        f.visititems(visitor)


if __name__ == "__main__":
    main()
