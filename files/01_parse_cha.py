"""
compute_clan_metrics.py
=======================
Computes discourse metrics from raw AphasiaBank .cha transcripts
using the SAME formulas CLAN uses. The goal is RQ1 validation:
do our automatically-computed metrics agree with CLAN's output?

CLAN metrics we can validate (all 14/14 non-null in our ground truth):
  - mlu_words      : MLU in words  (CLAN: mlu +t*PAR +f, but word-count mode)
  - mlu_morphemes  : MLU in morphemes (CLAN: mlu +t*PAR +f using %mor)
  - ttr            : Type-Token Ratio = types / tokens (CLAN: freq +t*PAR)
  - total_tokens   : Total word tokens (CLAN: freq)
  - total_types    : Total unique word types = NDW (CLAN: freq)
  - n_utterances   : Utterance count used in MLU

KEY DESIGN: CLAN uses two different commands with different word handling:
  mlu +f  → uses the %mor tier → filled pauses NOT counted (not in %mor)
  freq    → uses the *PAR main tier → filled pauses ARE counted

Usage:
    python compute_clan_metrics.py
    python compute_clan_metrics.py --cha_dir data/raw --clan_csv data/processed/clan_metrics.csv
"""

import re
import argparse
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# REGEX PATTERNS (CHAT annotation types)
# =============================================================================

TIMESTAMP     = re.compile(r'\x15?\d+_\d+\x15?')
NONVERBAL     = re.compile(r'&=[a-zA-Z_:]+')
FRAGMENT      = re.compile(r'&\+[a-zA-Z]+')
ERROR_CODE    = re.compile(r'\[\*[^\]]*\]')
TRAILING      = re.compile(r'\+\.\.\.|\"\/\.')
QUOTED        = re.compile(r'\+"/\.?|\+"')
RETRACING     = re.compile(r'\[/{1,3}\]')
OVERLAP       = re.compile(r'\+[<>!,]|\+\.')
SPECIAL_FORM  = re.compile(r'@[a-zA-Z]+')
SCOPE         = re.compile(r'\[\+[^\]]*\]')
COMMENT       = re.compile(r'\[%[^\]]*\]')
UNCERTAIN     = re.compile(r'\[=\?([^\]]*)\]')
ANNOTATIONS   = re.compile(r'\[[^\]]*\]')
CORRECTION    = re.compile(r'\[:\s*([^\]]+)\]')
ANGLE_RETRACE = re.compile(r'<[^>]*>\s*\[/{1,3}\]')
ANGLE_OVERLAP = re.compile(r'[<>]')
FILLED_PAUSE  = re.compile(r'&-([a-zA-Z_]+)')
CROSS_SPEAKER = re.compile(r'&\*[A-Z]{2,3}:[a-zA-Z]*')


# =============================================================================
# CHAT CLEANING
# =============================================================================

def _clean(raw: str, keep_filled_pauses: bool) -> str:
    """
    Core cleaning function. Replicates CLAN's text pre-processing.

    keep_filled_pauses=False : for MLU  (matches CLAN mlu using %mor tier)
    keep_filled_pauses=True  : for TTR  (matches CLAN freq using *PAR tier)

    CLAN rule: filled pauses (&-uh, &-um) appear on the *PAR tier
    but are NOT in the %mor tier, so mlu does not count them.
    The freq command processes *PAR directly and DOES count them.
    """
    text = raw

    text = TIMESTAMP.sub(' ', text)
    text = re.sub(r'^\*[A-Z]{2,3}:\s*', '', text)
    text = CROSS_SPEAKER.sub(' ', text)
    text = ANGLE_RETRACE.sub(' ', text)       # <word> [/] → removed
    text = CORRECTION.sub(r' \1 ', text)      # [: word] → keep correction
    text = ERROR_CODE.sub(' ', text)
    text = SCOPE.sub(' ', text)
    text = COMMENT.sub(' ', text)
    text = UNCERTAIN.sub(r' \1 ', text)
    text = TRAILING.sub(' ', text)
    text = QUOTED.sub(' ', text)
    text = NONVERBAL.sub(' ', text)           # &=laughs → removed
    text = FRAGMENT.sub(' ', text)            # &+word → removed (both modes)

    if keep_filled_pauses:
        text = FILLED_PAUSE.sub(r'\1', text)  # &-uh → uh  (for freq/TTR)
    else:
        text = FILLED_PAUSE.sub(' ', text)    # &-uh → gone (for mlu)

    text = OVERLAP.sub(' ', text)
    text = RETRACING.sub(' __RETRACE__ ', text)
    text = SPECIAL_FORM.sub(' ', text)
    text = ANNOTATIONS.sub(' ', text)
    text = ANGLE_OVERLAP.sub(' ', text)
    text = re.sub(r'[.?!,;:]', ' ', text)

    # Remove the word immediately before each __RETRACE__ marker
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        if words[i] == '__RETRACE__':
            if out:
                out.pop()   # remove retrace word
        else:
            out.append(words[i])
        i += 1

    text = ' '.join(w.lower() for w in out if w.strip())
    text = re.sub(r'\bxxx\b|\byyy\b|\bwww\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_for_mlu(raw: str) -> str:
    """For MLU: no filled pauses (matches CLAN %mor tier)."""
    return _clean(raw, keep_filled_pauses=False)


def clean_for_freq(raw: str) -> str:
    """For TTR/tokens: filled pauses kept (matches CLAN freq command)."""
    return _clean(raw, keep_filled_pauses=True)


def should_exclude(raw: str) -> bool:
    """
    CLAN excludes utterances marked [+ exc], cross-speaker embedded turns,
    and utterances with no countable content after cleaning.
    """
    if '[+ exc]' in raw:
        return True
    if CROSS_SPEAKER.search(raw) and not re.search(r'\*PAR:', raw):
        # utterances where the whole content is a cross-speaker code
        stripped = CROSS_SPEAKER.sub('', raw)
        stripped = TIMESTAMP.sub('', stripped)
        stripped = re.sub(r'[.?!,\s]', '', stripped)
        if not stripped.strip():
            return True
    cleaned = clean_for_mlu(raw)
    return not cleaned.strip()


# =============================================================================
# MORPHEME COUNTING (CLAN manual Chapter 21 rules)
# =============================================================================

# Common irregular past tense forms — each = 1 extra morpheme (past)
IRREGULAR_PAST = {
    'went','came','saw','said','told','made','got','took','gave','knew',
    'thought','found','felt','kept','left','lost','put','ran','sat','stood',
    'heard','let','met','set','cut','hit','hurt','broke','chose','drove',
    'fell','forgot','grew','held','lay','led','meant','paid','rose','sang',
    'slept','spent','spread','stuck','swam','threw','wore','won','wrote',
    'caught','bought','brought','built','dealt','dug','drew','fought',
    'frozen','hung','shook','stolen','struck','swept','taught','torn',
}

# Irregular plurals — already plural, count +1 morpheme
IRREGULAR_PLURALS = {
    'children','men','women','feet','teeth','mice','geese','sheep',
    'people','oxen','lice','dice',
}

# Contractions: each clitic = +1 morpheme (CLAN manual rule)
CONTRACTIONS = [
    re.compile(r"\w+n't\b",   re.I),   # can't, don't, won't
    re.compile(r"\w+'ve\b",   re.I),   # I've, we've
    re.compile(r"\w+'ll\b",   re.I),   # I'll, he'll
    re.compile(r"\w+'d\b",    re.I),   # I'd, she'd
    re.compile(r"\w+'re\b",   re.I),   # they're, we're
    re.compile(r"\w+'m\b",    re.I),   # I'm
]

# Words to NOT add suffix morphemes to (common words that look like suffixes)
NOT_SUFFIX = {
    # -ed endings that are base forms
    'bed','red','led','fed','wed','shed','sped',
    # -er endings that are not comparatives
    'after','never','over','under','other','water','butter','father',
    'mother','sister','winter','summer','letter','matter','flower',
    'number','order','paper','river','silver','tiger','tower','wonder',
    'finger','center','enter','member','power','upper','super',
    # -est endings that are not superlatives
    'forest','interest','modest','honest','harvest',
}


def morphemes_per_word(word: str) -> int:
    """
    Count morphemes in one word using CLAN's rule-based approach.
    Returns 0 for non-words, minimum 1 for real words.
    """
    w = word.lower().strip()
    if not w or not re.search(r'[a-zA-Z]', w):
        return 0

    count = 1  # every word = 1 base morpheme

    # Contractions: matched first — return immediately
    for pat in CONTRACTIONS:
        if pat.fullmatch(w):
            return count + 1

    # Possessive 's not caught above
    if w.endswith("'s"):
        return count + 1

    # Words in exception list — no extra morphemes
    if w in NOT_SUFFIX:
        return count

    # Irregular past tense
    if w in IRREGULAR_PAST:
        return count + 1

    # Irregular plurals
    if w in IRREGULAR_PLURALS:
        return count + 1

    # Regular past tense -ed (min 5 chars to avoid "bed", "red")
    if w.endswith('ed') and len(w) >= 5:
        return count + 1

    # Progressive -ing (min 6 chars: "singing" not "king")
    if w.endswith('ing') and len(w) >= 6:
        return count + 1

    # Comparative -er
    if w.endswith('er') and len(w) >= 5 and w not in NOT_SUFFIX:
        return count + 1

    # Superlative -est
    if w.endswith('est') and len(w) >= 6 and w not in NOT_SUFFIX:
        return count + 1

    # Plural or 3sg -s (not -ss like "glass", "class", "dress")
    if w.endswith('s') and len(w) >= 3 and not w.endswith('ss'):
        return count + 1

    return count


def count_morphemes(clean_text: str) -> int:
    """Count total morphemes in a cleaned utterance string."""
    return sum(morphemes_per_word(w) for w in clean_text.split())


# =============================================================================
# FILE PARSING
# =============================================================================

def parse_cha_file(filepath: Path) -> dict:
    """
    Parse one .cha file and compute all CLAN-equivalent metrics.
    Returns one dict (one row in the output CSV).
    """
    filepath = Path(filepath)

    with open(filepath, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # ---- Metadata from @ID: PAR header ----
    participant_id = filepath.stem
    corpus = age = sex = diagnosis = wab_aq = None

    for line in lines:
        line = line.strip()
        if line.startswith('@ID:') and '|PAR|' in line:
            parts = line[4:].strip().split('|')
            # @ID: lang|corpus|PAR|age|sex|group|...|role|...|custom|
            corpus    = parts[1].strip() if len(parts) > 1 else None
            age_raw   = parts[3].strip() if len(parts) > 3 else None
            sex       = parts[4].strip() if len(parts) > 4 else None
            diagnosis = parts[5].strip() if len(parts) > 5 else None
            wab_raw   = parts[9].strip() if len(parts) > 9 else None

            if age_raw:
                m = re.match(r'(\d+);(\d+)', age_raw)
                if m:
                    age = int(m.group(1)) + int(m.group(2)) / 12.0
                else:
                    try:
                        age = float(re.sub(r'[^\d.]', '', age_raw))
                    except ValueError:
                        pass

            if wab_raw:
                try:
                    wab_aq = float(wab_raw)
                except ValueError:
                    pass
            break

    # ---- Collect raw *PAR: utterances ----
    raw_utterances = []
    in_par = False
    current = ''

    for line in lines:
        stripped = line.rstrip('\n')

        if stripped.startswith('*'):
            # Save previous PAR utterance
            if in_par and current.strip():
                raw_utterances.append(current.strip())
            in_par = stripped.startswith('*PAR:')
            current = stripped[5:].strip() if in_par else ''

        elif in_par and (stripped.startswith('\t') or stripped.startswith('    ')):
            # Continuation line — skip dependent tiers (%, @)
            inner = stripped.strip()
            if not inner.startswith('%') and not inner.startswith('@'):
                current += ' ' + inner

        elif stripped.startswith('@') or stripped.startswith('%'):
            # Section/tier markers — end current PAR utterance
            if in_par and current.strip():
                raw_utterances.append(current.strip())
                in_par = False
                current = ''

    # Don't forget last utterance
    if in_par and current.strip():
        raw_utterances.append(current.strip())

    # ---- Compute metrics ----
    # MLU uses clean_for_mlu (no filled pauses — matches CLAN %mor tier)
    # TTR uses clean_for_freq (with filled pauses — matches CLAN freq command)

    mlu_word_lens  = []
    mlu_morph_lens = []
    freq_tokens    = []   # for TTR, total_tokens, total_types

    for raw in raw_utterances:
        # --- MLU computation ---
        if not should_exclude(raw):
            mlu_clean = clean_for_mlu(raw)
            if mlu_clean.strip():
                words = mlu_clean.split()
                if words:
                    mlu_word_lens.append(len(words))
                    mlu_morph_lens.append(count_morphemes(mlu_clean))

        # --- Freq/TTR computation (ALL utterances, no exclusions like CLAN freq) ---
        freq_clean = clean_for_freq(raw)
        freq_tokens.extend(freq_clean.split())

    # ---- Aggregate ----
    n_utterances    = len(mlu_word_lens)
    total_morphemes = sum(mlu_morph_lens)
    total_tokens    = len(freq_tokens)
    total_types     = len(set(freq_tokens))

    mlu_words     = round(float(np.mean(mlu_word_lens)),  3) if mlu_word_lens  else None
    mlu_morphemes = round(float(np.mean(mlu_morph_lens)), 3) if mlu_morph_lens else None
    ttr           = round(total_types / total_tokens,     4) if total_tokens   else None

    return {
        'participant_id':       participant_id,
        'corpus':               corpus,
        'age':                  round(age, 2) if age is not None else None,
        'sex':                  sex,
        'diagnosis':            diagnosis,
        'wab_aq':               wab_aq,
        # Validated metrics (direct comparison to CLAN ground truth)
        'mlu_words_ours':       mlu_words,
        'mlu_morphemes_ours':   mlu_morphemes,
        'ttr_ours':             ttr,
        'total_tokens_ours':    total_tokens,
        'total_types_ours':     total_types,
        'n_utterances_ours':    n_utterances,
        'total_morphemes_ours': total_morphemes,
    }


# =============================================================================
# RUN ON DIRECTORY
# =============================================================================

def run(cha_dir: str, out_path: str) -> pd.DataFrame:
    cha_dir  = Path(cha_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cha_files = sorted(cha_dir.rglob('*.cha'))
    if not cha_files:
        print(f"No .cha files found in {cha_dir}")
        return pd.DataFrame()

    print(f"Found {len(cha_files)} .cha files. Computing metrics...\n")

    rows = []
    for f in cha_files:
        try:
            row = parse_cha_file(f)
            rows.append(row)
            print(
                f"  {row['participant_id']:20s} | "
                f"utts={row['n_utterances_ours']:4d} | "
                f"tokens={row['total_tokens_ours']:5d} | "
                f"types={row['total_types_ours']:4d} | "
                f"MLU-w={str(row['mlu_words_ours']):7s} | "
                f"MLU-m={str(row['mlu_morphemes_ours']):7s} | "
                f"TTR={str(row['ttr_ours']):6s}"
            )
        except Exception as e:
            import traceback
            print(f"  ERROR on {f.name}: {e}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows → {out_path}")
    return df


# =============================================================================
# VALIDATE AGAINST CLAN GROUND TRUTH (RQ1)
# =============================================================================

def validate(our_csv: str, clan_csv: str, out_dir: str):
    """
    Compare our metrics to CLAN ground truth.
    Pearson r, MAE, bias per metric.
    Produces scatter plots and a summary CSV.
    This directly answers RQ1.
    """
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_ours = pd.read_csv(our_csv)
    df_clan = pd.read_csv(clan_csv)

    print(f"\nOur metrics:  {len(df_ours)} participants")
    print(f"CLAN metrics: {len(df_clan)} participants")

    df = df_ours.merge(df_clan, on='participant_id', how='inner')
    print(f"Matched:      {len(df)} participants\n")

    if len(df) == 0:
        print("ERROR: No matching participant IDs. Check filenames match CLAN output IDs.")
        return None

    # (our_col, clan_col, display_name)
    pairs = [
        ('mlu_words_ours',     'mlu_words_clan',   'MLU in words'),
        ('mlu_morphemes_ours', 'mlu_morphemes',     'MLU in morphemes'),
        ('ttr_ours',           'ttr_clan',          'Type-Token Ratio'),
        ('total_tokens_ours',  'total_tokens_clan', 'Total tokens'),
        ('total_types_ours',   'total_types_clan',  'Total types (NDW)'),
        ('n_utterances_ours',  'n_utterances_mlu',  'N utterances (MLU)'),
    ]

    results = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    print("=" * 78)
    print(f"{'Metric':<22} {'N':>4} {'Pearson r':>10} {'p-value':>10} "
          f"{'MAE':>8} {'Bias':>8} {'r≥0.70?':>10}")
    print("=" * 78)

    for i, (our_col, clan_col, label) in enumerate(pairs):
        if our_col not in df.columns or clan_col not in df.columns:
            print(f"{label:<22}  column missing (our={our_col}, clan={clan_col})")
            continue

        paired = df[[our_col, clan_col]].dropna()
        n = len(paired)
        if n < 3:
            print(f"{label:<22}  only {n} paired rows, skipping")
            continue

        x = paired[clan_col].values   # CLAN = ground truth
        y = paired[our_col].values    # ours = computed

        r, p   = stats.pearsonr(x, y)
        mae    = float(np.mean(np.abs(y - x)))
        bias   = float(np.mean(y - x))
        passed = '✓ YES' if r >= 0.70 else '✗ NO'

        results.append({
            'metric':     label,
            'n':          n,
            'pearson_r':  round(float(r), 4),
            'p_value':    round(float(p), 4),
            'mae':        round(mae, 4),
            'bias':       round(bias, 4),
            'validated':  bool(r >= 0.70),
        })

        print(f"{label:<22} {n:>4} {r:>10.4f} {p:>10.4f} {mae:>8.3f} "
              f"{bias:>+8.3f} {passed:>10}")

        # Scatter plot
        ax = axes[i]
        ax.scatter(x, y, alpha=0.75, s=65, color='steelblue',
                   edgecolors='white', linewidth=0.5, zorder=3)
        lims = [min(x.min(), y.min()) * 0.93, max(x.max(), y.max()) * 1.07]
        ax.plot(lims, lims, 'k--', lw=1.2, label='Perfect agreement', zorder=2)
        m_reg, b_reg = np.polyfit(x, y, 1)
        xr = np.linspace(lims[0], lims[1], 100)
        ax.plot(xr, m_reg * xr + b_reg, 'r-', lw=1.8,
                label=f'Our fit  r={r:.3f}', zorder=4)
        ax.set_xlabel(f'CLAN  {label}', fontsize=9)
        ax.set_ylabel(f'Ours  {label}', fontsize=9)
        ax.set_title(f'{label}\nr={r:.3f}  MAE={mae:.3f}  bias={bias:+.3f}',
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(lims); ax.set_ylim(lims)

    print("=" * 78)

    df_results = pd.DataFrame(results)
    val_csv    = out_dir / 'rq1_validation_results.csv'
    df_results.to_csv(val_csv, index=False)

    plt.suptitle(
        f'RQ1 Validation: Our Metrics vs CLAN Ground Truth  (N={len(df)} participants)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plot_path = out_dir / 'rq1_validation_scatter.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Summary
    n_pass = df_results['validated'].sum() if len(df_results) > 0 else 0
    print(f"\nValidated {n_pass}/{len(df_results)} metrics at r ≥ 0.70")

    print(f"\nMetrics VALIDATED (r ≥ 0.70) → enter reward function:")
    for _, row in df_results[df_results['validated']].iterrows():
        print(f"  ✓  {row['metric']:<28}  r={row['pearson_r']:.3f}")

    print(f"\nMetrics NOT validated → used with caution / noted as limitation:")
    for _, row in df_results[~df_results['validated']].iterrows():
        print(f"  ✗  {row['metric']:<28}  r={row['pearson_r']:.3f}")

    print(f"\nSaved → {val_csv}")
    print(f"Saved → {plot_path}")
    return df_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute CLAN-equivalent metrics from .cha files and validate (RQ1)'
    )
    parser.add_argument('--cha_dir',      default='data/raw')
    parser.add_argument('--out',          default='data/processed/our_metrics.csv')
    parser.add_argument('--clan_csv',     default='data/processed/clan_metrics.csv')
    parser.add_argument('--val_dir',      default='results/validation')
    parser.add_argument('--compute_only', action='store_true')
    args = parser.parse_args()

    df_ours = run(cha_dir=args.cha_dir, out_path=args.out)

    if not args.compute_only and Path(args.clan_csv).exists():
        validate(our_csv=args.out, clan_csv=args.clan_csv, out_dir=args.val_dir)
    elif not args.compute_only:
        print(f"\nNo CLAN ground truth at {args.clan_csv} — skipping validation.")
        print("Run 02_clan_metrics.py first, then rerun this script.")