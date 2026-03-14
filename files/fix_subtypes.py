# fix_subtypes.py
import pandas as pd

df = pd.read_csv("data/processed/parsed_transcripts.csv")

df["subtype"] = df["subtype"].str.strip()
df["subtype"] = df["subtype"].replace({
    "Control":        "control",
    "NotAphasicByWab": "NotAphasicByWAB",
})

df.to_csv("data/processed/parsed_transcripts.csv", index=False)
print(df["subtype"].value_counts().to_string())
print("Done.")