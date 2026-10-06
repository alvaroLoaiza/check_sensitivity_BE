"""
Step 4 - statistics on features.csv (all rows kept, proton-transfer rows included).

Everything is done separately for Li+ and NH4+.

correlations.csv   Spearman rho of each feature with BE, a 95% CI from a bootstrap
                   that resamples whole MOLECULES (sites of one molecule are not
                   independent), the partial rho with monomer net charge removed,
                   and rho within the net-charge -1 monomers only.
models.csv         Ridge regression per ion with EVERY feature, scored leave-one-molecule-out
                   (R2, MAE, MAE on proton-transfer rows vs the rest), plus, for each
                   feature family: the family alone, and the full model without it.
family_summary.csv Per family: R2 alone, and R2 lost when it is removed from the full model.
predictions.csv    The held-out prediction of every row for every model.
shap_features.csv  SHAP of every row and feature for the full model (kJ/mol); exact for a
                   linear model: phi_j = w_j * (x_j - mean x_j).
shap_families.csv  The same summed per family, per row.
shap_summary.csv   Mean |SHAP| per family.
*.png              With --plots: correlation forest plot, family plot, parity plot of the
                   full model, family SHAP strip.

Families: net charge (control); (1) local charge: V, Efield, Qnet at every radius,
V_all, Efield_all, nearest_atom_charge, ion_charge_nbo; (2) proximity: contact and
nearest-atom distances, atoms per shell, distance to and count of every group type,
contact-group type; (3) monomer orbitals: HOMO-2..LUMO+2, the three gaps, their
contact-group shares; (4) ion LUMO: the cross gap; proton transfer: the flag and the
longest N-H. Within one ion, (4) is -HOMO plus a constant, so it cannot be told apart
from (3).

Complex-derived descriptors include the polarization and charge transfer that
binding itself causes, so a strong correlation here shows they DESCRIBE binding,
not that they can predict it from the bare monomer.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 3, 13)
TARGET = "be_kjmol"
MIN_ROWS = 15            # skip a feature/ion with fewer non-missing rows than this

_FIXED = ["monomer_charge", "ion_charge_nbo", "V_all", "Efield_all", "nearest_atom_dist",
          "nearest_atom_charge", "contact_dist", "homo_mon_eV", "gap_cross_eV",
          "homo_mon_frac_contact", "proton_transfer", "ion_NH_max"]
_PREFIXES = ("V_r", "Efield_r", "Qnet_r", "n_env_r", "mon_", "dist_", "count_")

# Every feature in features.csv belongs to one family. The model questions are
# asked per family, because features inside a family (e.g. V/E/Q at several radii)
# overlap too much to be separated one by one with ~55 rows.
FAMILY_ORDER = ["net charge", "(1) local charge", "(2) proximity", "(3) monomer orbitals",
                "(4) ion LUMO", "proton transfer"]
DIST_FILL = 10.0   # Angstrom; a group type the molecule does not have counts as "far away"


def family_of(col):
    if col == "monomer_charge":
        return "net charge"
    if col.startswith(("V_", "Efield_", "Qnet_")) or col in ("nearest_atom_charge", "ion_charge_nbo"):
        return "(1) local charge"
    if (col in ("contact_dist", "nearest_atom_dist") or col.startswith(("n_env_r", "dist_", "count_"))
            or col.startswith("contact_group=")):
        return "(2) proximity"
    if col.startswith("mon_") or col in ("homo_mon_eV", "homo_mon_frac_contact"):
        return "(3) monomer orbitals"
    if col == "gap_cross_eV":
        return "(4) ion LUMO"
    if col in ("proton_transfer", "ion_NH_max"):
        return "proton transfer"
    return None


def design_matrix(g):
    """
    All model features for one ion's rows: numeric columns that have a family,
    contact_group one-hot encoded, dist_<group> gaps filled with DIST_FILL,
    and columns that are constant (or entirely empty) for this ion dropped.
    Returns (X DataFrame, {column: family}).
    """
    x = g.copy()
    if "contact_group" in x:
        x = pd.concat([x, pd.get_dummies(x["contact_group"], prefix="contact_group", prefix_sep="=",
                                         dtype=float)], axis=1)
    cols = [c for c in x.columns if family_of(c) and pd.api.types.is_numeric_dtype(x[c])]
    x = x[cols].astype(float)
    dist = [c for c in cols if c.startswith("dist_")]
    x[dist] = x[dist].fillna(DIST_FILL)
    x = x.loc[:, x.notna().any() & (x.nunique(dropna=True) > 1)]
    left = x.columns[x.isna().any()].tolist()
    if left:
        raise ValueError("missing values in %s; give them a fill rule" % ", ".join(left))
    return x, {c: family_of(c) for c in x.columns}


def feature_list(df):
    cols = [c for c in _FIXED if c in df.columns]
    cols += [c for c in df.columns if c.startswith(_PREFIXES) and c not in cols]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def _spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return spearmanr(x, y)[0]


def _boot_ci(x, y, groups, n_boot, rng):
    mols = np.unique(groups)
    idx = {m: np.flatnonzero(groups == m) for m in mols}
    vals = []
    for _ in range(n_boot):
        s = np.concatenate([idx[m] for m in rng.choice(mols, len(mols))])
        vals.append(_spearman(x[s], y[s]))
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return (np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan))


def _partial(x, y, z):
    """Spearman partial correlation of x and y with z removed (ranks, linear residuals)."""
    a = np.column_stack([np.ones(len(z)), rankdata(z)])
    res = lambda v: v - a @ np.linalg.lstsq(a, v, rcond=None)[0]   # noqa: E731
    rx, ry = res(rankdata(x)), res(rankdata(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0, 1]


def correlations(df, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ion, g in df.groupby("ion"):
        for f in feature_list(df):
            m = g[f].notna().values
            if m.sum() < MIN_ROWS:
                continue
            x = g[f].values[m].astype(float)
            y = g[TARGET].values[m].astype(float)
            grp = g["mol"].values[m]
            lo, hi = _boot_ci(x, y, grp, n_boot, rng)
            q = g["monomer_charge"].values[m]
            w = q == -1
            rows.append(dict(
                ion=ion, feature=f, n=int(m.sum()), rho=_spearman(x, y), ci_lo=lo, ci_hi=hi,
                ci_excludes_0=bool(lo > 0 or hi < 0),
                partial_rho_net_charge=np.nan if f == "monomer_charge" else _partial(x, y, q),
                n_charge_m1=int(w.sum()),
                rho_charge_m1=np.nan if f == "monomer_charge" or w.sum() < MIN_ROWS else _spearman(x[w], y[w])))
    return pd.DataFrame(rows)


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def _score(name, ion, x, g, kind, family):
    y = g[TARGET].values
    p = cross_val_predict(_ridge(), x.values, y, groups=g["mol"].values, cv=LeaveOneGroupOut())
    err = p - y
    pt = g["proton_transfer"].values == 1 if "proton_transfer" in g else np.zeros(len(g), bool)
    metric = dict(ion=ion, model=name, kind=kind, family=family, n_features=x.shape[1], n_rows=len(g),
                  R2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2), MAE=np.mean(np.abs(err)),
                  MAE_mean_baseline=np.mean(np.abs(y - y.mean())),
                  MAE_proton_transfer=np.mean(np.abs(err[pt])) if pt.any() else np.nan,
                  MAE_other_rows=np.mean(np.abs(err[~pt])))
    pred = pd.DataFrame(dict(label=g["label"].values, mol=g["mol"].values, ion=ion, model=name,
                             proton_transfer=pt.astype(int), be_kjmol=y, be_pred=p, residual=y - p))
    return metric, pred


def models(df):
    """
    Per ion, leave-one-molecule-out ridge models:
      full                       every feature
      full without ion_charge_nbo  (charge transfer is partly the binding itself)
      (1) local charge alone without ion_charge_nbo
      <family> alone             only that family's features
      full without <family>      R2 lost when the family is removed = what it adds
    """
    metrics, preds = [], []
    for ion, g in df.groupby("ion"):
        g = g.dropna(subset=[TARGET])
        x, fam = design_matrix(g)
        runs = [("full", "full", None, x)]
        if "ion_charge_nbo" in x:
            runs.append(("full without ion_charge_nbo", "check", "(1) local charge",
                         x.drop(columns="ion_charge_nbo")))
        local = [c for c in x.columns if fam[c] == "(1) local charge" and c != "ion_charge_nbo"]
        if "ion_charge_nbo" in x and local:
            runs.append(("(1) local charge alone without ion_charge_nbo", "check", "(1) local charge",
                         x[local]))
        for f in FAMILY_ORDER:
            cols = [c for c in x.columns if fam[c] == f]
            if not cols:
                continue
            runs.append(("%s alone" % f, "alone", f, x[cols]))
            if len(cols) < x.shape[1]:
                runs.append(("full without %s" % f, "removed", f, x.drop(columns=cols)))
        for name, kind, f, xx in runs:
            m, p = _score(name, ion, xx, g, kind, f)
            metrics.append(m)
            preds.append(p)
    metrics = pd.DataFrame(metrics)
    full = metrics[metrics["model"] == "full"].set_index("ion")["R2"]
    metrics["R2_lost_vs_full"] = np.where(metrics["kind"] == "removed",
                                          metrics["ion"].map(full) - metrics["R2"], np.nan)
    return metrics, pd.concat(preds, ignore_index=True)


def family_summary(metrics):
    """One row per (ion, family): R2 alone, R2 lost when removed."""
    a = metrics[metrics["kind"] == "alone"].set_index(["ion", "family"])["R2"].rename("R2_alone")
    r = metrics[metrics["kind"] == "removed"].set_index(["ion", "family"])["R2_lost_vs_full"]
    return pd.concat([a, r], axis=1).reset_index()


def shap_full(df):
    """
    Exact SHAP of the full ridge model (fitted on all rows of each ion):
    phi_j = w_j * (x_j - mean x_j); a row's phis add up to (prediction - mean prediction).
    Returns per-feature values, per-family values (sum over the family's features),
    and a summary of mean |SHAP| per family. Within a family the split between
    overlapping features is arbitrary; the family totals are the meaningful part.
    """
    feat_rows, fam_rows, summ = [], [], []
    for ion, g in df.groupby("ion"):
        g = g.dropna(subset=[TARGET])
        x, fam = design_matrix(g)
        model = _ridge().fit(x.values, g[TARGET].values)
        sc, rg = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
        w = rg.coef_ / sc.scale_
        phi = pd.DataFrame((x.values - sc.mean_) * w, columns=x.columns, index=g.index)
        keys = dict(label=g["label"].values, ion=ion, be_kjmol=g[TARGET].values,
                    pred_kjmol=model.predict(x.values), base_kjmol=rg.intercept_,
                    proton_transfer=g["proton_transfer"].values if "proton_transfer" in g else 0)
        feat_rows.append(pd.concat([pd.DataFrame(keys, index=g.index), phi.add_prefix("shap_")], axis=1))
        fams = [f for f in FAMILY_ORDER if any(fam[c] == f for c in x.columns)]
        fphi = pd.DataFrame({f: phi[[c for c in x.columns if fam[c] == f]].sum(axis=1) for f in fams})
        fam_rows.append(pd.concat([pd.DataFrame(keys, index=g.index), fphi], axis=1))
        for f in fams:
            summ.append(dict(ion=ion, family=f, n_features=sum(fam[c] == f for c in x.columns),
                             mean_abs_shap_kjmol=fphi[f].abs().mean(), alpha=rg.alpha_))
    return (pd.concat(feat_rows, ignore_index=True), pd.concat(fam_rows, ignore_index=True),
            pd.DataFrame(summ))
