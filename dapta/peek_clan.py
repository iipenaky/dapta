import pandas as pd

filepath = "data/clan/Baycrest9336a.eval.xls"

print("=== TRYING EXCEL ENGINES ===")
for engine in ("xlrd", "openpyxl"):
    try:
        df = pd.read_excel(filepath, header=None, engine=engine)
        print(f"Engine '{engine}' worked. Shape: {df.shape}")
        print(df.head(20).to_string())
        break
    except Exception as e:
        print(f"Engine '{engine}' failed: {e}")

print("\n=== RAW TEXT CONTENT ===")
try:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 30:
                break
            print(f"Line {i}: {repr(line)}")
except Exception as e:
    print(f"Text read failed: {e}")