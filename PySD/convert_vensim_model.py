from pathlib import Path
import pysd

#import sys
#print(sys.executable)
#print(sys.version)

def translate_vensim_to_pysd(mdl_path: str, out_dir: str) -> Path:
    mdl_path = Path(mdl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mdl_path.exists():
        raise FileNotFoundError(f"MDL file not found: {mdl_path}")

    # 出力ファイル名（.mdl -> .py）
    out_py = out_dir / (mdl_path.stem + ".py")

    # PySDでVensimモデルを読み込み（必要に応じて初回は変換が走る）
    model = pysd.read_vensim(str(mdl_path))

    # 変換済みPythonモデルを書き出し
    model.export(str(out_py))

    return out_py


if __name__ == "__main__":
    mdl = r"C:\Users\tomus\Documents\GitHub\river_management\River_management_xls_to3.mdl"
    out = r"C:\Users\tomus\Documents\GitHub\river_management\PySD"

    out_py = translate_vensim_to_pysd(mdl, out)
    print(f"Saved translated PySD model to: {out_py}")


