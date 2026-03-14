"""
02_clan_metrics.py
==================
Phase 2: Parse CLAN output files → ground truth CSV for validation.

Handles your exact CLAN output formats:
  .mlu.cex  → "Ratio of morphemes over utterances = X"
  .frq.cex  → TTR, total tokens, total types (plain text)
  .vocd.cex → raw lemma text (no D value — VOCD computed in Phase 3)
  .eval.xls → real XML spreadsheet format

Usage:
    python 02_clan_metrics.py
"""

import re
import yaml
import logging
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

with open("config.yaml") as f:
    CFG = yaml.safe_load(f)

CLAN_DIR = Path(CFG["paths"]["clan_outputs"])
OUT_DIR  = Path(CFG["paths"]["processed"])
OUT_DIR.mkdir(parents=True, exist_ok=True)
C        = CFG["clan"]

logging.basicConfig(
    filename=Path(CFG["paths"]["logs"]) / "02_clan_metrics.log",
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)


def read_text(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


# ── MLU parser ────────────────────────────────────────────────────────────────
def parse_mlu(path):
    """
    Your format:
      Number of: utterances = 98, morphemes = 591
      Ratio of morphemes over utterances = 6.031
    """
    text = read_text(path)
    out  = {"mlu_morphemes": None, "n_utterances_mlu": None}

    # Ratio = MLU in morphemes
    m = re.search(r"Ratio of morphemes over utterances\s*=\s*([\d.]+)", text)
    if m:
        out["mlu_morphemes"] = float(m.group(1))

    # Utterance count
    m = re.search(r"utterances\s*=\s*(\d+)", text)
    if m:
        out["n_utterances_mlu"] = int(m.group(1))

    # Morpheme count
    m = re.search(r"morphemes\s*=\s*(\d+)", text)
    if m:
        out["total_morphemes"] = int(m.group(1))

    return out


# ── FREQ parser ───────────────────────────────────────────────────────────────
def parse_freq(path):
    text = read_text(path)
    out  = {"ttr_clan": None, "total_tokens_clan": None, "total_types_clan": None}

    m = re.search(r"([\d.]+)\s+Type/Token ratio", text, re.I)
    if m: out["ttr_clan"] = float(m.group(1))

    m = re.search(r"(\d+)\s+Total number of items \(tokens\)", text, re.I)
    if m: out["total_tokens_clan"] = int(m.group(1))

    m = re.search(r"(\d+)\s+Total number of different item types", text, re.I)
    if m: out["total_types_clan"] = int(m.group(1))

    return out


# ── VOCD parser ───────────────────────────────────────────────────────────────
def parse_vocd(path):
    """
    Your VOCD file contains raw lemmatized text (not a D value).
    We store the lemma text so Phase 3 can compute VOCD-D from it,
    which is more accurate anyway than CLAN's older implementation.
    """
    text = read_text(path)

    # Skip header lines (contain CLAN command info)
    lines = text.splitlines()
    content_lines = []
    header_done = False
    for line in lines:
        if "********" in line:
            header_done = True
            continue
        if header_done and line.strip():
            content_lines.append(line.strip())

    lemma_text = " ".join(content_lines)
    return {"vocd_lemma_text": lemma_text if lemma_text else None}


# ── EVAL parser (real XML) ────────────────────────────────────────────────────
def parse_eval_xml(path):
    """
    Your .eval.xls is a proper XML spreadsheet.
    Parses all columns from the header row, extracts values for PAR speaker.
    """
    out = {
        "total_utts_clan":   None,
        "mlu_words_clan":    None,
        "mlu_utts_clan":     None,
        "ndw_clan":          None,
        "nuw_clan":          None,
        "maze_words_clan":   None,
        "unintelligible_clan": None,
        "total_words_clan":  None,
    }

    try:
        tree = ET.parse(path)
        root = tree.getroot()

        # Handle XML namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Find all rows
        rows = []
        for elem in root.iter(f"{ns}Row"):
            row_data = []
            for cell in elem.findall(f"{ns}Cell"):
                data = cell.find(f"{ns}Data")
                row_data.append(data.text if data is not None else "")
            rows.append(row_data)

        if len(rows) < 2:
            return out

        # First row is header
        headers = [str(h).strip() if h else "" for h in rows[0]]

        # Find PAR row (Role = Participant or Code = PAR)
        par_row = None
        role_idx = next((i for i, h in enumerate(headers)
                         if "Role" in h or "role" in h), None)
        code_idx = next((i for i, h in enumerate(headers)
                         if h.upper() == "CODE"), None)

        for row in rows[1:]:
            if role_idx is not None and role_idx < len(row):
                if "participant" in str(row[role_idx]).lower():
                    par_row = row
                    break
            if code_idx is not None and code_idx < len(row):
                if str(row[code_idx]).upper() == "PAR":
                    par_row = row
                    break

        # Fallback: use last data row
        if par_row is None and len(rows) > 1:
            par_row = rows[-1]

        if par_row is None:
            return out

        def get_val(col_name, as_type=float):
            """Find column by partial name match and return value."""
            for i, h in enumerate(headers):
                if col_name.lower() in h.lower() and i < len(par_row):
                    try:
                        return as_type(par_row[i])
                    except (ValueError, TypeError):
                        return None
            return None

        out["total_utts_clan"]     = get_val("Total_Utts",  int)
        out["mlu_words_clan"]      = get_val("MLU_Words",   float)
        out["mlu_utts_clan"]       = get_val("MLU_Utts",    float)
        out["ndw_clan"]            = get_val("NDW",         int)
        out["nuw_clan"]            = get_val("NUW",         int)
        out["maze_words_clan"]     = get_val("Maze",        int)
        out["unintelligible_clan"] = get_val("Unintel",     int)
        out["total_words_clan"]    = get_val("Total_Words", int)

        # Also try alternative column names
        if out["mlu_words_clan"] is None:
            out["mlu_words_clan"]  = get_val("MLU_W",  float)
        if out["ndw_clan"] is None:
            out["ndw_clan"]        = get_val("Diff_W", int)

    except ET.ParseError as e:
        logging.warning(f"XML parse error {path.name}: {e}")
    except Exception as e:
        logging.warning(f"EVAL parse error {path.name}: {e}")

    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.rule("[bold cyan]Phase 2: Parsing CLAN outputs")

    mlu_files = sorted(CLAN_DIR.glob(f"*{C['mlu_extension']}"))
    if not mlu_files:
        console.print(f"[red]No {C['mlu_extension']} files found in {CLAN_DIR}[/red]")
        return

    pids = [f.name.replace(C["mlu_extension"], "") for f in mlu_files]
    console.print(f"Found [bold]{len(pids)}[/bold] participants\n")

    rows    = []
    missing = []

    for pid in pids:
        row = {"participant_id": pid}

        for ext, parser in [
            (C["mlu_extension"],  parse_mlu),
            (C["freq_extension"], parse_freq),
            (C["vocd_extension"], parse_vocd),
            (C["eval_extension"], parse_eval_xml),
        ]:
            path = CLAN_DIR / f"{pid}{ext}"
            if path.exists():
                row.update(parser(path))
            else:
                missing.append(f"{pid}{ext}")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Print what we actually got
    table = Table(title="CLAN Parsing Summary")
    table.add_column("Column",   style="cyan")
    table.add_column("Non-null", style="green")
    table.add_column("Missing",  style="red")
    for col in df.columns:
        if col == "vocd_lemma_text":
            nn = int(df[col].notna().sum())
            table.add_row(col + " (text)", str(nn), str(len(df) - nn))
        else:
            nn = int(df[col].notna().sum())
            table.add_row(col, str(nn), str(len(df) - nn))
    console.print(table)

    out_path = OUT_DIR / "clan_metrics.csv"
    df.to_csv(out_path, index=False)

    if missing:
        console.print(f"\n[yellow]{len(missing)} files not found (first 3):[/yellow]")
        for f in missing[:3]:
            console.print(f"  - {f}")

    console.print(f"\n[green]✓ Saved → {out_path}[/green]")

    # Show a sample of what was extracted
    show_cols = [c for c in ["participant_id", "mlu_words_clan", "ttr_clan",
                               "ndw_clan", "maze_words_clan", "total_utts_clan"]
                 if c in df.columns]
    console.print("\nSample (first 3 rows):")
    console.print(df[show_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()