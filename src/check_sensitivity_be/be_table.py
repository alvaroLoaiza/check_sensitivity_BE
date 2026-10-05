"""Load all_BE_values.xlsx into a tidy table: one row per (monomer site, ion)."""
import pandas as pd

from .labels import ION_COLUMNS, parse_label


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
                             be_kjmol=r[col] if pd.notna(r[col]) else float("nan")))
    return pd.DataFrame(rows)
