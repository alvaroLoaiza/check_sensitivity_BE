"""
Usage (from the repo root, after `py -m pip install -e .`):

    py -m check_sensitivity_be manifest --root "C:\\...\\LONI_work_folder"
    py -m check_sensitivity_be verify   --manifest manifest.csv

`manifest` only looks at folder names (fast, does not download OneDrive files).
`verify` opens the logs and proves the mapping is right.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .be_table import load_be_table
from .discover import build_manifest, summarize
from .energies import binding_energy_kjmol, read_log_info

DEFAULT_BE = Path(__file__).resolve().parents[2] / "data" / "all_BE_values.xlsx"
BE_TOL_KJMOL = 0.05  # table values are rounded to ~0.001 kJ/mol; allow slack


def cmd_manifest(args):
    be = load_be_table(args.be)
    manifest = build_manifest(args.root, be)
    manifest.to_csv(args.output, index=False)
    print(summarize(manifest))
    print("\nWrote %s" % args.output)


def cmd_verify(args):
    manifest = pd.read_csv(args.manifest, keep_default_na=False, na_values=[""])
    todo = manifest[manifest["category"].isin(["usable", "logs_but_no_be"])]
    print("Verifying %d rows (reading each log once; ~1 s per 30 MB log)..." % len(todo),
          file=sys.stderr)

    cache = {}

    def info(path):
        if path not in cache:
            cache[path] = read_log_info(path)
        return cache[path]

    out = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        print("  [%d/%d] %s %s" % (i, len(todo), r.label, r.ion), file=sys.stderr)
        c, m, n = info(r.complex_log), info(r.monomer_log), info(r.ion_log)
        row = dict(label=r.label, ion=r.ion, has_be=r.has_be, be_table=r.be_kjmol)
        if None in (c.energy_ha, m.energy_ha, n.energy_ha):
            row["be_calc"] = float("nan")
        else:
            row["be_calc"] = binding_energy_kjmol(c, m, n)
        row["d_be"] = row["be_calc"] - r.be_kjmol if r.has_be else float("nan")
        row["charge_ok"] = (c.charge is not None and None not in (m.charge, n.charge)
                            and c.charge == m.charge + n.charge)
        row["natoms_ok"] = (c.natoms is not None and None not in (m.natoms, n.natoms)
                            and c.natoms == m.natoms + n.natoms)
        row["terminated_ok"] = c.normal_termination and m.normal_termination and n.normal_termination
        row.update(monomer_charge=m.charge, complex_charge=c.charge,
                   n_atoms_monomer=m.natoms, n_atoms_complex=c.natoms)
        out.append(row)

    res = pd.DataFrame(out)
    res.to_csv(args.output, index=False)

    have = res[res["has_be"] == True]  # noqa: E712
    good_be = have[have["d_be"].abs() <= BE_TOL_KJMOL]
    print("\nMapping checks over %d rows:" % len(res))
    print("  charge balance   (complex = monomer + ion): %d ok" % res["charge_ok"].sum())
    print("  atom balance     (complex = monomer + ion): %d ok" % res["natoms_ok"].sum())
    print("  normal termination (all three logs)       : %d ok" % res["terminated_ok"].sum())
    print("  BE recomputed vs table (within %.2f kJ/mol): %d of %d"
          % (BE_TOL_KJMOL, len(good_be), len(have)))

    bad = res[~(res["charge_ok"] & res["natoms_ok"] & res["terminated_ok"])
              | (res["has_be"] & (res["d_be"].abs() > BE_TOL_KJMOL))]
    if len(bad):
        print("\nRows that failed at least one check:")
        print(bad[["label", "ion", "be_table", "be_calc", "d_be",
                   "charge_ok", "natoms_ok", "terminated_ok"]].to_string(index=False))
    new = res[~res["has_be"] & res["be_calc"].notna()]
    if len(new):
        print("\nBE values computable from logs but missing from the table:")
        print(new[["label", "ion", "be_calc"]].to_string(index=False))
    print("\nWrote %s" % args.output)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_sensitivity_be", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="map every BE-table row to its three logs")
    p.add_argument("--root", required=True, help="LONI_work_folder (the folder holding 000.NH4_p, 050.*, ...)")
    p.add_argument("--be", default=str(DEFAULT_BE), help="BE spreadsheet (default: data/all_BE_values.xlsx)")
    p.add_argument("-o", "--output", default="manifest.csv")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("verify", help="prove the mapping: charge/atom balance and BE recomputation")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("-o", "--output", default="verify.csv")
    p.set_defaults(func=cmd_verify)

    args = ap.parse_args(argv)
    args.func(args)
