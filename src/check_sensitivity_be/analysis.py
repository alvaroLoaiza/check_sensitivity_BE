"""
Step 4 - statistics on features.csv (all rows kept, proton-transfer rows included).

Binding (analysis/):  one target, BE, separately for Li+ and NH4+.
Selectivity (analysis/selectivity/):  one row per site that has BOTH ions, with
    dBE      = BE_Li - BE_NH4   (> 0: NH4+ binds more strongly; the table's Diff)
    affinity = -(BE_Li + BE_NH4) / 2   (how strongly the site binds either ion)
  Features of the two complexes are kept side by side (suffix @Li / @NH4);
  monomer features, identical for both, appear once.

For every target:
  correlations   Spearman rho of each feature, 95% CI from a bootstrap that resamples
                 whole MOLECULES, partial rho with net charge removed, rho within net
                 charge -1. Every feature is reported; n < 15 is flagged low_n.
  models         Ridge regression, leave-one-molecule-out: every feature together, each
                 family alone, and every feature except one family (what it adds). Errors
                 are also split into proton-transfer rows and the rest.
  SHAP           Exact for the linear model: phi_j = w_j * (x_j - mean x_j), summed per
                 family. The split inside a family is arbitrary; family totals are not.

Families: net charge (control); (1) local charge: V, Efield, Qnet at every radius,
V_all, Efield_all, nearest_atom_charge, ion_charge_nbo; (2) proximity: contact and
nearest-atom distances, atoms per shell, distance to and count of every group type,
contact-group type; (3) monomer orbitals: HOMO-2..LUMO+2, the three gaps, their
contact-group shares; (4) ion LUMO: the cross gap; proton transfer: the flag and the
longest N-H. Within one ion, (4) is -HOMO plus a constant, so it cannot be told apart
from (3).

Complex-derived descriptors include the polarization and charge transfer that
binding itself causes, so a strong correlation shows they DESCRIBE binding, not
that they can predict it from the bare monomer.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 3, 13)
TARGET = "be_kjmol"
MIN_ROWS = 5         # below this a correlation is not computed at all
LOW_N = 15           # below this it is computed but flagged low_n
DIST_FILL = 10.0     # Angstrom; a group type the molecule does not have counts as "far away"
RT_KJMOL = 8.314462618e-3 * 298.15
PAIR = "Li-NH4"      # the 'ion' value of selectivity rows

FAMILY_ORDER = ["net charge", "(1) local charge", "(2) proximity", "(3) monomer orbitals",
                "(4) ion LUMO", "proton transfer"]


def base_name(col):
    """'V_all@Li' -> 'V_all';  'contact_group@NH4=ether(C-O-C)' -> 'contact_group'."""
    return col.split("@")[0].split("=")[0]


def family_of(col):
    b = base_name(col)
    if b == "monomer_charge":
        return "net charge"
    if b.startswith(("V_", "Efield_", "Qnet_")) or b in ("nearest_atom_charge", "ion_charge_nbo"):
        return "(1) local charge"
    if (b in ("contact_dist", "nearest_atom_dist", "contact_group")
            or b.startswith(("n_env_r", "dist_", "count_"))):
        return "(2) proximity"
    if b.startswith("mon_") or b in ("homo_mon_eV", "homo_mon_frac_contact"):
        return "(3) monomer orbitals"
    if b == "gap_cross_eV":
        return "(4) ion LUMO"
    if b in ("proton_transfer", "ion_NH_max"):
        return "proton transfer"
    return None


def feature_list(df):
    """Numeric feature columns, in family order."""
    cols = [c for c in df.columns if family_of(c) and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(cols, key=lambda c: FAMILY_ORDER.index(family_of(c)))


def design_matrix(g):
    """
    Model features for one group of rows: numeric family columns, contact-group type
    one-hot encoded, dist_<group> gaps filled with DIST_FILL, and columns that are
    constant or empty in these rows dropped. Returns (X, {column: family}).
    """
    x = g.copy()
    for c in [c for c in x.columns if base_name(c) == "contact_group" and "=" not in c]:
        x = pd.concat([x, pd.get_dummies(x[c], prefix=c, prefix_sep="=", dtype=float)], axis=1)
    cols = [c for c in x.columns if family_of(c) and pd.api.types.is_numeric_dtype(x[c])]
    x = x[cols].astype(float)
    dist = [c for c in cols if base_name(c).startswith("dist_")]
    x[dist] = x[dist].fillna(DIST_FILL)
    x = x.loc[:, x.notna().any() & (x.nunique(dropna=True) > 1)]
    left = x.columns[x.isna().any()].tolist()
    if left:
        raise ValueError("missing values in %s; give them a fill rule" % ", ".join(left))
    return x, {c: family_of(c) for c in x.columns}


# ----------------------------------------------------------------- correlations
def _spearman(x, y):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    with warnings.catch_warnings():          # small bootstrap resamples can be (nearly) constant
        warnings.simplefilter("ignore")
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
    return np.percentile(vals, [2.5, 97.5]) if len(vals) else (np.nan, np.nan)


def _partial(x, y, z):
    """Spearman partial correlation of x and y with z removed (ranks, linear residuals)."""
    a = np.column_stack([np.ones(len(z)), rankdata(z)])
    res = lambda v: v - a @ np.linalg.lstsq(a, v, rcond=None)[0]   # noqa: E731
    rx, ry = res(rankdata(x)), res(rankdata(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return np.nan
    return np.corrcoef(rx, ry)[0, 1]


def correlations(df, target=TARGET, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ion, g in df.groupby("ion"):
        for f in feature_list(df):
            m = (g[f].notna() & g[target].notna()).values
            if m.sum() < MIN_ROWS:
                continue
            x = g[f].values[m].astype(float)
            y = g[target].values[m].astype(float)
            if np.std(x) == 0:
                continue
            lo, hi = _boot_ci(x, y, g["mol"].values[m], n_boot, rng)
            q = g["monomer_charge"].values[m]
            w = q == -1
            rows.append(dict(
                ion=ion, target=target, family=family_of(f), feature=f, n=int(m.sum()),
                low_n=bool(m.sum() < LOW_N), rho=_spearman(x, y), ci_lo=lo, ci_hi=hi,
                ci_excludes_0=bool(lo > 0 or hi < 0),
                partial_rho_net_charge=np.nan if family_of(f) == "net charge" else _partial(x, y, q),
                n_charge_m1=int(w.sum()),
                rho_charge_m1=(np.nan if family_of(f) == "net charge" or w.sum() < MIN_ROWS
                               else _spearman(x[w], y[w]))))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- models
def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))


def _score(name, ion, x, g, kind, family, target):
    y = g[target].values
    p = cross_val_predict(_ridge(), x.values, y, groups=g["mol"].values, cv=LeaveOneGroupOut())
    err = p - y
    pt = g["proton_transfer"].values == 1 if "proton_transfer" in g else np.zeros(len(g), bool)
    metric = dict(ion=ion, target=target, model=name, kind=kind, family=family,
                  n_features=x.shape[1], n_rows=len(g),
                  R2=1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2), MAE=np.mean(np.abs(err)),
                  MAE_mean_baseline=np.mean(np.abs(y - y.mean())),
                  MAE_proton_transfer=np.mean(np.abs(err[pt])) if pt.any() else np.nan,
                  MAE_other_rows=np.mean(np.abs(err[~pt])))
    pred = pd.DataFrame(dict(label=g["label"].values, mol=g["mol"].values, ion=ion, target=target,
                             model=name, proton_transfer=pt.astype(int), observed=y, predicted=p,
                             residual=y - p))
    return metric, pred


def models(df, target=TARGET):
    """
    Per ion (or per pair), leave-one-molecule-out ridge models:
      full                          every feature
      full without ion_charge_nbo   (charge transfer is partly the binding itself)
      (1) local charge alone without ion_charge_nbo
      <family> alone                only that family
      full without <family>         what the family adds that the others cannot replace
    """
    metrics, preds = [], []
    for ion, g in df.groupby("ion"):
        g = g.dropna(subset=[target])
        x, fam = design_matrix(g)
        runs = [("full", "full", None, x)]
        icn = [c for c in x.columns if base_name(c) == "ion_charge_nbo"]
        if icn:
            runs.append(("full without ion_charge_nbo", "check", "(1) local charge", x.drop(columns=icn)))
            local = [c for c in x.columns if fam[c] == "(1) local charge" and c not in icn]
            if local:
                runs.append(("(1) local charge alone without ion_charge_nbo", "check",
                             "(1) local charge", x[local]))
        for f in FAMILY_ORDER:
            cols = [c for c in x.columns if fam[c] == f]
            if not cols:
                continue
            runs.append(("%s alone" % f, "alone", f, x[cols]))
            if len(cols) < x.shape[1]:
                runs.append(("full without %s" % f, "removed", f, x.drop(columns=cols)))
        for name, kind, f, xx in runs:
            m, p = _score(name, ion, xx, g, kind, f, target)
            metrics.append(m)
            preds.append(p)
    metrics = pd.DataFrame(metrics)
    full = metrics[metrics["model"] == "full"].set_index("ion")
    rem = metrics["kind"] == "removed"
    metrics["R2_lost_vs_full"] = np.where(rem, metrics["ion"].map(full["R2"]) - metrics["R2"], np.nan)
    for part in ("proton_transfer", "other_rows"):
        col = "MAE_" + part
        metrics["MAE_added_%s" % part] = np.where(rem, metrics[col] - metrics["ion"].map(full[col]), np.nan)
    return metrics, pd.concat(preds, ignore_index=True)


def family_summary(metrics):
    """Per (ion, family): R2 alone; R2 lost when removed; MAE added on transfer rows / the rest."""
    a = metrics[metrics["kind"] == "alone"].set_index(["ion", "family"])["R2"].rename("R2_alone")
    r = metrics[metrics["kind"] == "removed"].set_index(["ion", "family"])[
        ["R2_lost_vs_full", "MAE_added_proton_transfer", "MAE_added_other_rows"]]
    return pd.concat([a, r], axis=1).reset_index()


# ------------------------------------------------------------------------- SHAP
def shap_full(df, target=TARGET):
    """
    Exact SHAP of the full ridge model fitted on all rows of each ion (or pair).
    Returns per-feature values, per-family values (summed), a summary of mean |SHAP|
    per family, and, per family, the feature with the largest mean |SHAP| (with its
    values) so plots can colour each family by a real feature.
    """
    feat_rows, fam_rows, summ = [], [], []
    for ion, g in df.groupby("ion"):
        g = g.dropna(subset=[target])
        x, fam = design_matrix(g)
        model = _ridge().fit(x.values, g[target].values)
        sc, rg = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
        w = rg.coef_ / sc.scale_
        phi = pd.DataFrame((x.values - sc.mean_) * w, columns=x.columns, index=g.index)
        keys = dict(label=g["label"].values, ion=ion, target=target, observed=g[target].values,
                    pred=model.predict(x.values), base=rg.intercept_,
                    proton_transfer=g["proton_transfer"].values if "proton_transfer" in g else 0)
        feat_rows.append(pd.concat([pd.DataFrame(keys, index=g.index), phi.add_prefix("shap_"),
                                    x.add_prefix("value_")], axis=1))
        fams = [f for f in FAMILY_ORDER if any(fam[c] == f for c in x.columns)]
        fphi = pd.DataFrame({f: phi[[c for c in x.columns if fam[c] == f]].sum(axis=1) for f in fams})
        lead = {}
        for f in fams:
            cols = [c for c in x.columns if fam[c] == f]
            top = max(cols, key=lambda c: phi[c].abs().mean())
            lead[f] = top
            fphi["lead_value:" + f] = x[top].values
            summ.append(dict(ion=ion, target=target, family=f, n_features=len(cols),
                             mean_abs_shap_kjmol=fphi[f].abs().mean(), lead_feature=top, alpha=rg.alpha_))
        fam_rows.append(pd.concat([pd.DataFrame(keys, index=g.index), fphi], axis=1))
    return (pd.concat(feat_rows, ignore_index=True), pd.concat(fam_rows, ignore_index=True),
            pd.DataFrame(summ))


# ------------------------------------------------------------------ selectivity
def pair_frame(df):
    """
    One row per site with both ions. Columns that are identical for the two ions
    (monomer features) appear once; the rest appear as <col>@Li and <col>@NH4.
    Adds dBE = BE_Li - BE_NH4 and affinity = -(BE_Li + BE_NH4) / 2, in kJ/mol.
    """
    li = df[df["ion"] == "Li"].set_index("label")
    nh = df[df["ion"] == "NH4"].set_index("label")
    labels = li.index.intersection(nh.index)
    li, nh = li.loc[labels], nh.loc[labels]
    cols = {"mol": li["mol"], "site": li["site"]}
    for c in df.columns:
        if c in ("label", "mol", "site", "ion", "errors", TARGET, "proton_transfer") or c not in li:
            continue
        a, b = li[c], nh[c]
        if a.equals(b) or ((a.isna() & b.isna()) | (a == b)).all():
            cols[c] = a
        else:
            cols[c + "@Li"] = a
            cols[c + "@NH4"] = b
    cols["be_Li"] = li[TARGET]
    cols["be_NH4"] = nh[TARGET]
    cols["dBE"] = li[TARGET] - nh[TARGET]
    cols["affinity"] = -(li[TARGET] + nh[TARGET]) / 2
    cols["proton_transfer"] = nh["proton_transfer"] if "proton_transfer" in nh else pd.Series(0, index=labels)
    cols["ion"] = pd.Series(PAIR, index=labels)
    out = pd.concat(cols, axis=1)
    out.index.name = "label"
    return out.copy().reset_index()


def per_molecule(df):
    """
    Per molecule and ion: the best (most negative) site BE and a Boltzmann-weighted
    BE over its sites at 298.15 K (w_i ~ exp(-BE_i / RT)); and the resulting
    selectivity dBE = BE_Li - BE_NH4 for both. With BE spreads of tens of kJ/mol the
    Boltzmann value is usually within a fraction of a kJ/mol of the best site.
    """
    rows = []
    for (mol, ion), g in df.dropna(subset=[TARGET]).groupby(["mol", "ion"]):
        be = g[TARGET].values
        w = np.exp(-(be - be.min()) / RT_KJMOL)
        w /= w.sum()
        best = g.loc[g[TARGET].idxmin()]
        rows.append(dict(mol=mol, ion=ion, n_sites=len(g), be_best=be.min(), best_site=best["label"],
                         be_boltzmann=float(np.sum(w * be)), be_mean=be.mean(),
                         proton_transfer_at_best=int(best.get("proton_transfer", 0) == 1)))
    long = pd.DataFrame(rows)
    m = long.pivot_table(index="mol", columns="ion",
                         values=["be_best", "be_boltzmann", "be_mean", "n_sites", "proton_transfer_at_best"])
    m.columns = ["%s_%s" % c for c in m.columns]
    sites = long.pivot(index="mol", columns="ion", values="best_site").add_prefix("best_site_")
    m = m.join(sites)
    for kind in ("best", "boltzmann", "mean"):
        if "be_%s_Li" % kind in m and "be_%s_NH4" % kind in m:
            m["dBE_%s" % kind] = m["be_%s_Li" % kind] - m["be_%s_NH4" % kind]
    return m.reset_index()
