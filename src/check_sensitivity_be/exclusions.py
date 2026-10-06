"""
Rows left out of the analysis on purpose, with the reason, in data/exclusions.csv
(columns: label, ion, reason). Kept as a file instead of deleting table rows so
the decision and its reason stay on record. `verify` still checks these rows and
lists them separately; `features` skips them.
"""
from pathlib import Path

import pandas as pd

DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[2] / "data" / "exclusions.csv"


def load_exclusions(path=DEFAULT_EXCLUSIONS):
    """{(label, ion): reason}; empty if the file does not exist."""
    path = Path(path)
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = {"label", "ion", "reason"} - set(df.columns)
    if missing:
        raise ValueError("%s is missing column(s): %s" % (path, ", ".join(sorted(missing))))
    return {(r.label.strip(), r.ion.strip()): r.reason.strip() for r in df.itertuples(index=False)}
