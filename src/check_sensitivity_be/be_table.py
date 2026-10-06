"""Load all_BE_values.xlsx into a tidy table: one row per (monomer site, ion)."""
import pandas as pd

from .labels import ION_COLUMNS, parse_label

_MINUS = str.maketrans({"\u2212": "-", "\u2013": "-", "\u2014": "-"})


def _to_float(value, label, col):
    """
    Cell value -> float (NaN if empty). Text cells are accepted when they hold a
    number (Excel 'number stored as text'); a typographic minus such as U+2212,
    which is easy to paste in by accident, is read as '-'. Anything else stops
    the run here, naming the cell, instead of failing later with a TypeError.
    """
    if pd.isna(value):
        return float("nan")
    if isinstance(value, str):
        text = value.translate(_MINUS).strip()
        if text == "":
            return float("nan")
        try:
            return float(text)
        except ValueError:
            raise ValueError("Non-numeric BE in the spreadsheet: label %s, column %s, value %r"
                             % (label, col, value)) from None
    return float(value)


def load_be_table(path):
    """
    Returns a DataFrame with columns
        label, mol, site, ion ('Li'/'NH4'), be_kjmol (NaN if no value)
    Every (label, ion) pair is kept, including the ones with no BE: a missing
    value means the singlep log did not exist when the table was made, and
    the manifest step uses that to cross-check the folder mapping.

    The spreadsheet's own 'BE_Li-NH4 Diff' column is ignored (it is
    BE_Li - BE_NH4 and can always be recomputed).
    """
    raw = pd.read_excel(path)
    raw = raw.rename(columns={raw.columns[0]: "label"})
    missing = [c for c in ION_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError("Columns not found in %s: %s" % (path, missing))

    rows = []
    for _, r in raw.iterrows():
        if pd.isna(r["label"]):
            continue
        lab = parse_label(r["label"])
        for col, ion in ION_COLUMNS.items():
            rows.append(dict(label=lab.raw, mol=lab.mol, site=lab.site, ion=ion,
                             be_kjmol=_to_float(r[col], lab.raw, col)))
    return pd.DataFrame(rows)
