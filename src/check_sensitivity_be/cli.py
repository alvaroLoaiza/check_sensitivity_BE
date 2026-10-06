"""
Usage (from the repo root, after `py -m pip install -e .`):

    py -m check_sensitivity_be manifest --root "C:\\...\\LONI_work_folder"
    py -m check_sensitivity_be verify   --manifest manifest.csv
    py -m check_sensitivity_be features --manifest manifest.csv
    py -m check_sensitivity_be analyze  --features features.csv --plots

`manifest` only looks at folder names (fast, does not download OneDrive files).
`verify` opens the logs and proves the mapping is right.
`features` writes one descriptor row per (site, ion) to features.csv.
`analyze` writes correlations, leave-one-molecule-out models and SHAP to analysis/.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from .be_table import load_be_table
from .discover import build_manifest, summarize
from .energies import binding_energy_kjmol, geometry_status, opt_logs_for, read_last_nre, read_log_info
from .exclusions import DEFAULT_EXCLUSIONS, load_exclusions

DEFAULT_BE = Path(__file__).resolve().parents[2] / "data" / "all_BE_values.xlsx"
BE_TOL_KJMOL = 0.05  # table values are rounded to ~0.001 kJ/mol; allow slack
_GEOM_RANK = {"ok": 0, "no_opt_log": 1, "mismatch": 2}


def cmd_manifest(args):
    be = load_be_table(args.be)
    manifest = build_manifest(args.root, be)
    manifest.to_csv(args.output, index=False)
    print(summarize(manifest))
    print("\nWrote %s" % args.output)


def _read_manifest(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def cmd_verify(args):
    manifest = _read_manifest(args.manifest)
    excluded = load_exclusions(args.exclusions)
    todo = manifest[manifest["category"].isin(["usable", "logs_but_no_be"])]
    print("Verifying %d rows (reading each log once; ~1 s per 30 MB log)..." % len(todo),
          file=sys.stderr)

    cache, opt_cache = {}, {}

    def info(path):
        if path not in cache:
            cache[path] = read_log_info(path)
        return cache[path]

    def geom(sp_info):
        nres = []
        for p in opt_logs_for(sp_info.path):
            if p not in opt_cache:
                opt_cache[p] = read_last_nre(p)
            nres.append(opt_cache[p])
        return geometry_status(sp_info.nre_ha, nres)

    out = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        print("  [%d/%d] %s %s" % (i, len(todo), r.label, r.ion), file=sys.stderr)
        c, m, n = info(r.complex_log), info(r.monomer_log), info(r.ion_log)
        row = dict(label=r.label, ion=r.ion, has_be=r.has_be, be_table=r.be_kjmol,
                   excluded=(r.label, r.ion) in excluded)
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
        # complex must list the monomer atoms first and the ion atoms last
        row["order_ok"] = bool(c.atomic_numbers) and c.atomic_numbers == m.atomic_numbers + n.atomic_numbers
        statuses = {k: geom(x) for k, x in (("complex", c), ("monomer", m), ("ion", n))}
        row["geom_status"] = max(statuses.values(), key=_GEOM_RANK.get)
        row["geom_detail"] = ",".join("%s:%s" % kv for kv in statuses.items() if kv[1] != "ok")
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
    print("  atom order       (monomer first, ion last): %d ok" % res["order_ok"].sum())
    print("  normal termination (all three logs)       : %d ok" % res["terminated_ok"].sum())
    print("  singlep on final opt geometry (all three) : %d ok, %d no opt log, %d mismatch"
          % tuple((res["geom_status"] == s).sum() for s in ("ok", "no_opt_log", "mismatch")))
    print("  BE recomputed vs table (within %.2f kJ/mol): %d of %d"
          % (BE_TOL_KJMOL, len(good_be), len(have)))

    failed = (~(res["charge_ok"] & res["natoms_ok"] & res["terminated_ok"] & res["order_ok"])
              | (res["geom_status"] == "mismatch")
              | (res["has_be"] & (res["d_be"].abs() > BE_TOL_KJMOL)))
    cols = ["label", "ion", "be_table", "be_calc", "d_be", "charge_ok", "natoms_ok",
            "order_ok", "terminated_ok", "geom_status", "geom_detail"]
    bad = res[failed & ~res["excluded"]]
    if len(bad):
        print("\nRows that failed at least one check:")
        print(bad[cols].to_string(index=False))
    exc = res[res["excluded"]]
    if len(exc):
        print("\nExcluded rows (%s), checked but not counted as failures:" % Path(args.exclusions).name)
        print(exc[cols].to_string(index=False))
    no_opt = res[(res["geom_status"] == "no_opt_log") & ~res["excluded"]]
    if len(no_opt):
        print("\nRows where a singlep could not be compared with an opt log (not a failure):")
        print(no_opt[["label", "ion", "geom_detail"]].to_string(index=False))
    new = res[~res["has_be"] & res["be_calc"].notna()]
    if len(new):
        print("\nBE values computable from logs but missing from the table:")
        print(new[["label", "ion", "be_calc"]].to_string(index=False))
    print("\nWrote %s" % args.output)


def cmd_features(args):
    from .features.extract import FragmentCache, extract_row, feature_columns

    manifest = _read_manifest(args.manifest)
    excluded = load_exclusions(args.exclusions)
    usable = manifest[manifest["category"] == "usable"]
    keep = [(r.label, r.ion) not in excluded for r in usable.itertuples(index=False)]
    todo = usable[keep]
    print("Extracting features for %d rows (%d usable, %d excluded)..."
          % (len(todo), len(usable), len(usable) - len(todo)), file=sys.stderr)

    cache = FragmentCache()
    rows = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        print("  [%d/%d] %s %s" % (i, len(todo), r.label, r.ion), file=sys.stderr)
        mono = cache.get(r.monomer_log, want_homo_coeffs=True)   # cached: read once per monomer
        feat = extract_row(r.complex_log, r.monomer_log, r.ion_log, cache, args.radii)
        rows.append(dict(label=r.label, mol=r.mol, site=r.site, ion=r.ion,
                         monomer_charge=mono.charge, **feat))

    keys = ["label", "mol", "site", "ion", "monomer_charge"]
    df = pd.DataFrame(rows, columns=keys + feature_columns(args.radii))
    df.to_csv(args.output, index=False, float_format="%.6g")

    errs = df[df["errors"].fillna("") != ""]
    print("\n%d rows x %d columns; %d row(s) with extraction notes" % (len(df), df.shape[1], len(errs)))
    print("Contact group counts:")
    print(df.groupby(["ion", "contact_group"]).size().to_string())
    pt = df[df["proton_transfer"] == 1]
    if len(pt):
        print("\nNH4+ proton moved onto the monomer (ion H closer to a monomer N/O than to its own N):")
        print(pt[["label", "ion", "monomer_charge", "ion_charge_nbo", "ion_NH_max", "contact_group",
                  "be_kjmol"]].to_string(index=False))
    if len(errs):
        print("\nRows with extraction notes:")
        print(errs[["label", "ion", "errors"]].to_string(index=False))
    print("\nWrote %s" % args.output)


def cmd_analyze(args):
    try:
        from . import analysis as an
    except ImportError as e:
        sys.exit("analyze needs the analysis extras: py -m pip install -e \".[analysis]\"  (%s)" % e)
    df = pd.read_csv(args.features)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    print("%s: %d rows (%s), %d molecules; all rows kept, proton-transfer rows included"
          % (args.features, len(df), ", ".join("%s %d" % kv for kv in df["ion"].value_counts().sort_index().items()),
             df["mol"].nunique()))

    print("Correlations (%d bootstrap resamples of whole molecules)..." % args.n_boot, file=sys.stderr)
    corr = an.correlations(df, n_boot=args.n_boot, seed=args.seed)
    corr.to_csv(out / "correlations.csv", index=False)
    print("\nStrongest correlations with BE (95% CI excludes 0), per ion:")
    for ion, c in corr.groupby("ion"):
        c = c[c["ci_excludes_0"]].reindex(c["rho"].abs().sort_values(ascending=False).index).dropna(subset=["n"])
        print("  %s" % ion)
        print(c.head(args.top)[["feature", "rho", "ci_lo", "ci_hi", "partial_rho_net_charge", "rho_charge_m1"]]
              .round(2).to_string(index=False))

    print("Models...", file=sys.stderr)
    metrics, preds = an.models(df)
    metrics.to_csv(out / "models.csv", index=False)
    preds.to_csv(out / "predictions.csv", index=False)
    fam = an.family_summary(metrics)
    fam.to_csv(out / "family_summary.csv", index=False)

    print("\nFull model, every feature (leave-one-molecule-out):")
    full = metrics[metrics["kind"].isin(["full", "check"])]
    print(full[["ion", "model", "n_features", "R2", "MAE", "MAE_proton_transfer", "MAE_other_rows"]]
          .round(2).to_string(index=False))
    print("\nPer family: R2 with the family alone, and R2 lost when it is removed from the full model:")
    print(fam.pivot_table(index="family", columns="ion", values=["R2_alone", "R2_lost_vs_full"])
          .reindex([f for f in an.FAMILY_ORDER if f in set(fam["family"])]).round(2).to_string())

    pt = preds[(preds["proton_transfer"] == 1) & (preds["model"] == "full")]
    if len(pt):
        print("\nProton-transfer rows, full model (residual < 0: binds more strongly than predicted):")
        print(pt[["label", "ion", "be_kjmol", "be_pred", "residual"]].round(1).to_string(index=False))

    shap_f, shap_fam, shap_sum = an.shap_full(df)
    shap_f.to_csv(out / "shap_features.csv", index=False)
    shap_fam.to_csv(out / "shap_families.csv", index=False)
    shap_sum.to_csv(out / "shap_summary.csv", index=False)
    print("\nSHAP of the full model, mean |SHAP| per family (kJ/mol):")
    print(shap_sum.pivot_table(index="family", columns="ion", values="mean_abs_shap_kjmol")
          .reindex([f for f in an.FAMILY_ORDER if f in set(shap_sum["family"])]).round(1).to_string())

    if args.plots:
        try:
            from . import plots
        except ImportError as e:
            sys.exit("--plots needs matplotlib: py -m pip install -e \".[analysis]\"  (%s)" % e)
        plots.correlation_forest(corr, out / "correlations.png", top=args.top_plot)
        plots.families(fam, an.FAMILY_ORDER, out / "families.png")
        plots.parity(preds, metrics, "full", out / "parity_full.png")
        plots.family_shap_strip(shap_fam, an.FAMILY_ORDER, out / "shap_families.png")
        print("\nPlots written to %s" % out)
    print("\nWrote correlations, models, family_summary, predictions, shap_* CSVs in %s" % out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="check_sensitivity_be", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("manifest", help="map every BE-table row to its three logs")
    p.add_argument("--root", required=True, help="LONI_work_folder (the folder holding 000.NH4_p, 050.*, ...)")
    p.add_argument("--be", default=str(DEFAULT_BE), help="BE spreadsheet (default: data/all_BE_values.xlsx)")
    p.add_argument("-o", "--output", default="manifest.csv")
    p.set_defaults(func=cmd_manifest)

    p = sub.add_parser("verify", help="prove the mapping: balances, atom order, geometry, BE recomputation")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--exclusions", default=str(DEFAULT_EXCLUSIONS))
    p.add_argument("-o", "--output", default="verify.csv")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("features", help="one descriptor row per (site, ion) -> features.csv")
    p.add_argument("--manifest", default="manifest.csv")
    p.add_argument("--exclusions", default=str(DEFAULT_EXCLUSIONS))
    p.add_argument("--radii", type=float, nargs="+", default=[3.0, 4.0, 5.0],
                   help="charge-shell cutoffs in Angstrom (default 3 4 5)")
    p.add_argument("-o", "--output", default="features.csv")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("analyze", help="correlations, leave-one-molecule-out models, SHAP -> analysis/")
    p.add_argument("--features", default="features.csv")
    p.add_argument("--out", default="analysis", help="output folder (default: analysis)")
    p.add_argument("--n-boot", type=int, default=2000, help="molecule bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=8, help="correlations to print per ion")
    p.add_argument("--top-plot", type=int, default=25, help="features in the correlation plot")
    p.add_argument("--plots", action="store_true", help="also write PNG plots (needs matplotlib)")
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args(argv)
    args.func(args)
