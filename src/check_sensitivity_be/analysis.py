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


# ------------------------------------------------------------- functional groups
from .features.groups import CANONICAL_GROUP_TYPES, safe_name  # noqa: E402

_PRETTY = {safe_name(t): t for t in CANONICAL_GROUP_TYPES}
GROUP_MIN_MOL = 3          # a group's effect is estimated only if it occurs in >= this many molecules
REF_GROUP = "sulfonate(SO3)"


def _contact_col(df):
    return "contact_group" if "contact_group" in df.columns else None


def group_contact_table(df, target=TARGET, group_col="contact_group"):
    """Per ion and contact group: sites, molecules, median/min/max of the target, net charges present."""
    rows = []
    for ion, g in df.dropna(subset=[target, group_col]).groupby("ion"):
        for grp, h in g.groupby(group_col):
            rows.append(dict(ion=ion, target=target, contact_group=grp, n_sites=len(h), n_molecules=h["mol"].nunique(),
                             median=h[target].median(), min=h[target].min(), max=h[target].max(),
                             net_charges=",".join(str(int(v)) for v in sorted(h["monomer_charge"].unique())),
                             proton_transfer_sites=int((h.get("proton_transfer", 0) == 1).sum())))
    t = pd.DataFrame(rows)
    return t.sort_values(["ion", "median"]) if len(t) else t


def _ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _boot_molecules(g, fit, n_boot, rng):
    """Refit on molecule-bootstrap samples; fit returns a Series or None (sample unusable)."""
    mols = g["mol"].unique()
    idx = {m: np.flatnonzero(g["mol"].values == m) for m in mols}
    draws = []
    for _ in range(n_boot):
        s = np.concatenate([idx[m] for m in rng.choice(mols, len(mols))])
        r = fit(g.iloc[s])
        if r is not None:
            draws.append(r)
    return pd.DataFrame(draws)


def contact_group_effects(df, target=TARGET, group_col="contact_group", ref=REF_GROUP,
                          min_mol=GROUP_MIN_MOL, n_boot=2000, seed=0):
    """
    target ~ contact group (one-hot, relative to `ref`) + monomer net charge, per ion, on the
    sites whose contact group occurs in >= min_mol molecules. 95% intervals from refitting on
    molecule-bootstrap samples. Effect > 0: weaker binding than at `ref` (for BE).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for ion, g in df.dropna(subset=[target, group_col]).groupby("ion"):
        nm = g.groupby(group_col)["mol"].nunique()
        keep = sorted(nm[nm >= min_mol].index)
        if ref not in keep or len(keep) < 2:
            continue
        g = g[g[group_col].isin(keep)]
        others = [k for k in keep if k != ref]

        def fit(h):
            if set(h[group_col]) != set(keep) or h["monomer_charge"].nunique() < 2:
                return None
            X = np.column_stack([np.ones(len(h)), h["monomer_charge"].values] +
                                [(h[group_col] == k).values.astype(float) for k in others])
            return pd.Series(_ols(X, h[target].values)[1:], index=["net charge"] + others)

        est = fit(g)
        if est is None:
            continue
        bs = _boot_molecules(g, fit, n_boot, rng)
        for k in est.index:
            rows.append(dict(ion=ion, target=target, term=k, relative_to=ref if k != "net charge" else "per unit charge",
                             effect=est[k], ci_lo=bs[k].quantile(0.025), ci_hi=bs[k].quantile(0.975),
                             n_sites=int((g[group_col] == k).sum()) if k != "net charge" else len(g),
                             n_molecules=int(g.loc[g[group_col] == k, "mol"].nunique()) if k != "net charge"
                             else g["mol"].nunique(),
                             n_boot_used=len(bs)))
    return pd.DataFrame(rows)


def _dist_cols(df, suffix=""):
    return [c for c in df.columns if c.startswith("dist_") and c.endswith(suffix)
            and (suffix or "@" not in c)]


def group_distance_effects(df, target=TARGET, min_mol=GROUP_MIN_MOL, n_boot=2000, seed=0, suffix=""):
    """
    How a functional group affects the target as a function of its distance from the ion:
        target = b0 + gamma * net_charge + sum_g beta_g / d_g
    d_g = distance (Angstrom) from the ion to the nearest group of type g; 1/d_g = 0 when the
    molecule has no such group. beta_g / d is then the group's contribution at distance d
    (kJ/mol), the same 1/d decay as a point charge. Only groups present in >= min_mol molecules.
    Returns (coefficients with molecule-bootstrap intervals, partial residuals for plotting,
             bootstrap draws).
    """
    rng = np.random.default_rng(seed)
    coef_rows, resid_rows, draws_all = [], [], []
    for ion, g in df.dropna(subset=[target]).groupby("ion"):
        cols = [c for c in _dist_cols(g, suffix)
                if g.loc[g[c].notna(), "mol"].nunique() >= min_mol]
        if not cols:
            continue
        inv = pd.DataFrame({c: 1.0 / g[c] for c in cols}).fillna(0.0)
        names = ["net charge"] + cols

        def design(h_inv, h):
            return np.column_stack([np.ones(len(h)), h["monomer_charge"].values, h_inv.values])

        def fit(h):
            hi = pd.DataFrame({c: 1.0 / h[c] for c in cols}).fillna(0.0)
            if (hi == 0).all().any() or h["monomer_charge"].nunique() < 2:
                return None
            return pd.Series(_ols(design(hi, h), h[target].values)[1:], index=names)

        est = fit(g)
        if est is None:
            continue
        gb = g.reset_index(drop=True)
        bs = _boot_molecules(gb, fit, n_boot, rng)
        X = design(inv, g)
        b = _ols(X, g[target].values)
        fitted = X @ b
        for j, c in enumerate(cols):
            nsite = int(g[c].notna().sum())
            coef_rows.append(dict(ion=ion, target=target, group=_PRETTY.get(c.split("@")[0][5:], c.split("@")[0][5:]), column=c,
                                  beta_kjmol_A=est[c], ci_lo=bs[c].quantile(0.025), ci_hi=bs[c].quantile(0.975),
                                  effect_at_3A=est[c] / 3, effect_at_5A=est[c] / 5,
                                  n_sites_with_group=nsite, n_molecules_with_group=int(g.loc[g[c].notna(), "mol"].nunique()),
                                  n_boot_used=len(bs)))
            # partial residual: target minus everything except this group's term
            part = g[target].values - fitted + b[2 + j] * inv[c].values
            m = g[c].notna().values
            resid_rows.append(pd.DataFrame(dict(ion=ion, target=target, group=_PRETTY.get(c.split("@")[0][5:], c.split("@")[0][5:]), label=g["label"].values[m],
                                                distance=g[c].values[m], partial_residual=part[m],
                                                proton_transfer=(g["proton_transfer"].values[m] == 1).astype(int)
                                                if "proton_transfer" in g else 0)))
        coef_rows.append(dict(ion=ion, target=target, group="net charge", column="monomer_charge",
                              beta_kjmol_A=est["net charge"], ci_lo=bs["net charge"].quantile(0.025),
                              ci_hi=bs["net charge"].quantile(0.975), n_boot_used=len(bs)))
        d = bs.copy()
        d["ion"] = ion
        d["target"] = target
        draws_all.append(d)
    return (pd.DataFrame(coef_rows),
            pd.concat(resid_rows, ignore_index=True) if resid_rows else pd.DataFrame(),
            pd.concat(draws_all, ignore_index=True) if draws_all else pd.DataFrame())


def same_group_pairs(pairs):
    """Selectivity rows where Li+ and NH4+ touch the same group type, with that type as contact_group."""
    if "contact_group" in pairs.columns:
        return pairs.copy()
    a, b = pairs.get("contact_group@Li"), pairs.get("contact_group@NH4")
    if a is None or b is None:
        return pairs.iloc[0:0]
    out = pairs[a == b].copy()
    out["contact_group"] = a[a == b]
    return out


# ------------------------------------------------------------- fitted equations
def coefficients(df, target=TARGET):
    """
    The final model for each ion (or pair): the full ridge model fitted on ALL rows (the
    same fit the SHAP values come from). Written two ways, both exact:

      standardized:  y_hat = b0 + sum_j w_j * (x_j - mean_j) / sd_j      (w_j in kJ/mol per 1 SD)
      raw units:     y_hat = c0 + sum_j a_j * x_j                         (a_j = w_j / sd_j)

    with c0 = b0 - sum_j a_j * mean_j. Missing group distances are filled with DIST_FILL
    first, exactly as in the fit. Overlapping features share their weight arbitrarily,
    so individual weights are not effects; the equation as a whole is the prediction.
    """
    rows = []
    for ion, g in df.groupby("ion"):
        g = g.dropna(subset=[target])
        x, fam = design_matrix(g)
        model = _ridge().fit(x.values, g[target].values)
        sc, rg = model.named_steps["standardscaler"], model.named_steps["ridgecv"]
        a = rg.coef_ / sc.scale_
        c0 = rg.intercept_ - np.sum(a * sc.mean_)
        rows.append(dict(ion=ion, target=target, feature="(intercept)", family="", mean=np.nan, sd=np.nan,
                         weight_per_sd=rg.intercept_, weight_per_unit=c0, alpha=rg.alpha_, n_rows=len(g),
                         n_features=x.shape[1]))
        for j, c in enumerate(x.columns):
            rows.append(dict(ion=ion, target=target, feature=c, family=fam[c], mean=sc.mean_[j], sd=sc.scale_[j],
                             weight_per_sd=rg.coef_[j], weight_per_unit=a[j], alpha=rg.alpha_, n_rows=len(g),
                             n_features=x.shape[1]))
    out = pd.DataFrame(rows)
    out["abs_weight_per_sd"] = out["weight_per_sd"].abs().where(out["feature"] != "(intercept)")
    return out
