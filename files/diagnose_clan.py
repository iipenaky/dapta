# diagnose_clan.py
# Prints the first 40 lines of one of each file type so we can see the exact format

from pathlib import Path

clan_dir = Path("data/clan_outputs")

for ext in [".mlu.cex", ".frq.cex", ".vocd.cex", ".eval.xls"]:
    files = list(clan_dir.glob(f"*{ext}"))
    if not files:
        print(f"\n{'='*50}\nNo files found with extension: {ext}")
        continue
    
    filepath = files[0]
    print(f"\n{'='*50}")
    print(f"FILE: {filepath.name}")
    print(f"{'='*50}")
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i, line in enumerate(lines[:40]):
        print(f"{i+1:3d}  {line}", end="")