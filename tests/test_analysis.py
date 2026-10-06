"""analyze on a small synthetic features table: families, ablation, exact SHAP, outputs."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from check_sensitivity_be import analysis as an  # noqa: E402
from check_sensitivity_be import cli  # noqa: E402


def _synthetic(n_mol=8, sites=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    homos = rng.normal(-5.5, 0.3, n_mol)          # a monomer property: the same for both ions
    for ion, shift in (("Li", 0.0), ("NH4", -10.0)):
        for m in range(n_mol):
            q = [-1, -1, -2, 0][m % 4]
            homo = homos[m]
            for s in range(sites):
                v = rng.normal(-0.4 + 0.2 * q, 0.1)
                e = abs(rng.normal(0.2, 0.05))
                pt = int(ion == "NH4" and m == 2 and s == 0)
                rows.append(dict(
                    label="M%d_%d" % (m, s), mol="M%d" % m, site=s, ion=ion, monomer_charge=q,
                    V_all=v, V_r5=v + rng.normal(0, 0.02), Efield_r5=e, ion_charge_nbo=rng.normal(0.95, 0.02),
                    contact_dist=2.0 + 3 * e + rng.normal(0, 0.05),
                    contact_group=["sulfonate(SO3)", "ether(C-O-C)"][s % 2],
                    dist_halide=np.nan if m % 2 else rng.uniform(3, 6), count_halide=0 if m % 2 else 1,
                    homo_mon_eV=homo, gap_cross_eV=(0.7 if ion == "Li" else 0.3) - homo,
                    proton_transfer=pt, ion_NH_max=(1.03 + 0.7 * pt) if ion == "NH4" else np.nan,
                    be_kjmol=shift + 60 * v - 40 * e - 30 * pt + rng.normal(0, 1)))
    return pd.DataFrame(rows)


def test_every_feature_gets_a_family_and_gaps_are_filled():
    g = _synthetic()
    x, fam = an.design_matrix(g[g.ion == "NH4"])
    assert not x.isna().any().any()
    assert (x["dist_halide"] == an.DIST_FILL).any()
    assert fam["proton_transfer"] == "proton transfer" and fam["contact_dist"] == "(2) proximity"
    assert fam["contact_group=sulfonate(SO3)"] == "(2) proximity"
    xl, _ = an.design_matrix(g[g.ion == "Li"])
    assert "proton_transfer" not in xl and "ion_NH_max" not in xl     # constant / empty for Li


def test_correlations_report_every_feature_with_family_and_low_n():
    c = an.correlations(_synthetic(), n_boot=200)
    v = c[(c.feature == "V_all") & (c.ion == "Li")].iloc[0]
    assert v.rho > 0.5 and v.ci_excludes_0 and v.family == "(1) local charge" and not v.low_n
    h = c[(c.feature == "dist_halide") & (c.ion == "Li")].iloc[0]     # present in 12 of 24 rows
    assert h.n == 12 and h.low_n


def test_family_models_alone_and_removed():
    m, p = an.models(_synthetic())
    assert set(m.kind) == {"full", "check", "alone", "removed"}
    fam = an.family_summary(m)
    loc = fam[(fam.ion == "Li") & (fam.family == "(1) local charge")].iloc[0]
    assert loc.R2_alone > 0.5 and loc.R2_lost_vs_full > 0
    assert p.groupby("model").size().max() == len(_synthetic())
    assert {"MAE_added_proton_transfer", "MAE_added_other_rows"} <= set(fam.columns)


def test_shap_adds_up_to_the_prediction_and_families_sum_features():
    feats, fams, summ = an.shap_full(_synthetic())
    for ion in ("Li", "NH4"):
        f = feats[feats.ion == ion].dropna(axis=1, how="all")
        total = f[[c for c in f if c.startswith("shap_")]].sum(axis=1) + f["base"]
        assert np.allclose(total, f["pred"])
        g = fams[fams.ion == ion]
        fam_cols = [c for c in an.FAMILY_ORDER if c in g and g[c].notna().any()]
        assert np.allclose(g[fam_cols].sum(axis=1) + g["base"], g["pred"])
    assert "proton transfer" in set(summ[summ.ion == "NH4"].family)


def test_pair_frame_keeps_shared_columns_once_and_splits_the_rest():
    df = _synthetic()
    p = an.pair_frame(df)
    assert len(p) == 24 and (p.ion == an.PAIR).all()
    assert "homo_mon_eV" in p and "homo_mon_eV@Li" not in p          # same monomer for both ions
    assert {"V_all@Li", "V_all@NH4"} <= set(p.columns)
    row = df[(df.label == "M0_0")].set_index("ion").be_kjmol
    q = p[p.label == "M0_0"].iloc[0]
    assert q.dBE == pytest.approx(row["Li"] - row["NH4"])
    assert q.affinity == pytest.approx(-(row["Li"] + row["NH4"]) / 2)
    assert p.proton_transfer.sum() == 1
    m, _ = an.models(p, target="dBE")
    assert (m.target == "dBE").all()


def test_per_molecule_best_and_boltzmann():
    df = pd.DataFrame(dict(mol="X", ion=["Li", "Li", "NH4"], label=["X_1", "X_2", "X_1"],
                           be_kjmol=[-40.0, -10.0, -55.0], proton_transfer=0))
    m = an.per_molecule(df).iloc[0]
    assert m.be_best_Li == -40.0 and m.best_site_Li == "X_1"
    assert m.be_boltzmann_Li == pytest.approx(-40.0, abs=1e-3)      # 30 kJ/mol apart: all weight on X_1
    assert m.dBE_best == pytest.approx(15.0)


def test_analyze_command_writes_outputs(tmp_path):
    f = tmp_path / "features.csv"
    _synthetic().to_csv(f, index=False)
    args = ["analyze", "--features", str(f), "--out", str(tmp_path / "an"), "--n-boot", "100"]
    try:
        import matplotlib  # noqa: F401
        args.append("--plots")
    except ImportError:
        pass
    cli.main(args)
    for name in ("correlations.csv", "models.csv", "family_summary.csv", "predictions.csv",
                 "shap_features.csv", "shap_families.csv", "shap_summary.csv",
                 "selectivity/sites.csv", "selectivity/molecules.csv", "selectivity/models_dBE.csv",
                 "selectivity/correlations_affinity.csv"):
        assert (tmp_path / "an" / name).is_file(), name
    if "--plots" in args:
        for name in ("families.png", "correlations.png", "selectivity/affinity_map_sites.png",
                     "selectivity/affinity_map_molecules_boltzmann.png", "selectivity/shap_families_dBE.png"):
            assert (tmp_path / "an" / name).is_file(), name
