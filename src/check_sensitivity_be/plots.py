"""PNG plots for `analyze --plots` (matplotlib, no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ION_LABEL = {"Li": "Li$^+$", "NH4": "NH$_4^+$", "Li-NH4": "Li$^+$ vs NH$_4^+$"}
TARGET_LABEL = {"be_kjmol": "BE", "dBE": "dBE = BE$_{Li}$ - BE$_{NH4}$", "affinity": "affinity"}


def correlation_forest(corr, order, path):
    """
    Every feature, grouped by family (strongest |rho| first inside a family), one panel
    per ion. Filled: CI excludes 0. Grey: CI includes 0. Hollow: fewer than 15 sites.
    """
    ions = list(corr["ion"].unique())
    fams = [f for f in order if f in set(corr["family"])]
    blocks = {}
    for ion in ions:
        c = corr[(corr["ion"] == ion) & corr["rho"].notna()]
        blocks[ion] = [c[c["family"] == f].reindex(c[c["family"] == f]["rho"].abs()
                                                   .sort_values(ascending=False).index) for f in fams]
    n_max = max(sum(len(b) for b in blocks[i]) + 2 * len(fams) for i in ions)
    fig, axes = plt.subplots(1, len(ions), figsize=(7 * len(ions), 0.2 * n_max + 1.5), squeeze=False)
    for ax, ion in zip(axes[0], ions):
        y, ticks, labels = 0, [], []
        for f, b in zip(fams, blocks[ion]):
            if not len(b):
                continue
            ax.text(-0.98, y, f, fontsize=8, fontweight="bold", va="center")
            y += 1
            for _, r in b.iterrows():
                err = np.clip([[r["rho"] - r["ci_lo"]], [r["ci_hi"] - r["rho"]]], 0, None)
                ax.errorbar(r["rho"], y, xerr=err, fmt="none", ecolor="0.6", elinewidth=0.8, capsize=1.5)
                color = "tab:blue" if r["ci_excludes_0"] else "0.55"
                ax.scatter(r["rho"], y, s=18, zorder=3, edgecolors=color,
                           facecolors="none" if r["low_n"] else color)
                ticks.append(y)
                labels.append("%s (n=%d)" % (r["feature"], r["n"]) if r["low_n"] else r["feature"])
                y += 1
            y += 0.6
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_ylim(y, -1)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(-1, 1)
        t = corr[corr["ion"] == ion]["target"].iloc[0]
        ax.set_xlabel("Spearman rho with %s (95%% CI, molecule bootstrap)" % TARGET_LABEL.get(t, t))
        ax.set_title("%s  (grey: CI includes 0; hollow: n < 15)" % ION_LABEL.get(ion, ion), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def families(fam, order, path):
    """R2 alone; R2 lost when removed; MAE added on removal, transfer rows vs the rest."""
    ions = list(fam["ion"].unique())
    fams = [f for f in order if f in set(fam["family"])]
    panels = [("R2_alone", "R$^2$, family alone", None),
              ("R2_lost_vs_full", "R$^2$ lost, family removed", None)]
    if fam["MAE_added_proton_transfer"].notna().any():
        panels += [("MAE_added_other_rows", "MAE added when removed (kJ/mol)", "other rows"),
                   ("MAE_added_proton_transfer", "MAE added when removed (kJ/mol)", "proton-transfer rows")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 0.45 * len(fams) + 1.8), squeeze=False)
    for ax, (col, title, sub) in zip(axes[0], panels):
        w = 0.8 / len(ions)
        for k, ion in enumerate(ions):
            v = fam[fam["ion"] == ion].set_index("family").reindex(fams)[col]
            ax.barh(np.arange(len(fams)) + (k - (len(ions) - 1) / 2) * w, v, height=w,
                    label=ION_LABEL.get(ion, ion))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(fams)))
        ax.set_yticklabels(fams if ax is axes[0][0] else [""] * len(fams))
        ax.invert_yaxis()
        ax.set_title(title + ("\n" + sub if sub else ""), fontsize=9)
        ax.set_xlabel("leave-one-molecule-out", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parity(preds, metrics, model, path, units="kJ/mol"):
    """Held-out predicted vs observed; proton-transfer rows marked."""
    p = preds[preds["model"] == model]
    ions = list(p["ion"].unique())
    fig, axes = plt.subplots(1, len(ions), figsize=(5 * len(ions), 4.8), squeeze=False)
    for ax, ion in zip(axes[0], ions):
        g = p[p["ion"] == ion]
        pt = g["proton_transfer"] == 1
        ax.scatter(g.loc[~pt, "observed"], g.loc[~pt, "predicted"], s=22, label="sites")
        if pt.any():
            ax.scatter(g.loc[pt, "observed"], g.loc[pt, "predicted"], s=60, marker="^",
                       color="tab:red", label="proton transfer")
            for _, r in g[pt].iterrows():
                ax.annotate(r["label"], (r["observed"], r["predicted"]), fontsize=7,
                            xytext=(4, -10), textcoords="offset points")
        lo = min(g["observed"].min(), g["predicted"].min()) - 5
        hi = max(g["observed"].max(), g["predicted"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        m = metrics[(metrics["ion"] == ion) & (metrics["model"] == model)].iloc[0]
        t = TARGET_LABEL.get(m["target"], m["target"])
        ax.set_title("%s  R$^2$ = %.2f, MAE = %.1f %s" % (ION_LABEL.get(ion, ion), m["R2"], m["MAE"], units),
                     fontsize=10)
        ax.set_xlabel("%s from DFT (%s)" % (t, units))
        ax.set_ylabel("%s predicted, molecule held out (%s)" % (t, units))
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(model)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def family_shap_strip(shap_fam, summary, order, path):
    """
    Each site's summed SHAP per family, coloured by the value of the family's leading
    feature (named on the axis), so the direction of the effect is visible.
    Proton-transfer rows are triangles.
    """
    ions = list(shap_fam["ion"].unique())
    fig, axes = plt.subplots(1, len(ions), figsize=(7 * len(ions), 4.6), squeeze=False)
    fig.subplots_adjust(wspace=0.75)
    rng = np.random.default_rng(0)
    sc = None
    for ax, ion in zip(axes[0], ions):
        s = shap_fam[shap_fam["ion"] == ion]
        lead = summary[summary["ion"] == ion].set_index("family")["lead_feature"]
        fams = [f for f in order if f in s.columns and s[f].notna().any()]
        for k, f in enumerate(fams):
            v = s["lead_value:" + f].astype(float)
            z = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0 + 0.5
            pt = (s["proton_transfer"] == 1).values
            jit = k + rng.uniform(-0.2, 0.2, len(s))
            sc = ax.scatter(s[f][~pt], jit[~pt], c=z[~pt], cmap="coolwarm", vmin=0, vmax=1, s=16)
            ax.scatter(s[f][pt], jit[pt], c=z[pt], cmap="coolwarm", vmin=0, vmax=1, s=55,
                       marker="^", edgecolors="k", linewidths=0.6)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(["%s\n[colour: %s]" % (f, lead.get(f, "")) for f in fams], fontsize=7.5)
        ax.invert_yaxis()
        t = s["target"].iloc[0]
        ax.set_xlabel("summed SHAP of the family (kJ/mol), relative to the average site", fontsize=8)
        ax.set_title("%s, target %s  (triangles: proton transfer)" % (ION_LABEL.get(ion, ion),
                                                                    TARGET_LABEL.get(t, t)), fontsize=9)
    if sc is not None:
        cb = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.02)
        cb.set_label("value of the leading feature (low to high)", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def affinity_map(x_nh4, y_li, labels, pt, path, title, split=None, what="sites"):
    """|BE_NH4| vs |BE_Li| with the diagonal (equal binding); below it NH4+ binds more strongly."""
    x, y = np.abs(np.asarray(x_nh4, float)), np.abs(np.asarray(y_li, float))
    pt = np.asarray(pt, bool)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(x[~pt], y[~pt], s=24, label=what)
    if pt.any():
        ax.scatter(x[pt], y[pt], s=60, marker="^", color="tab:red", label="proton transfer (NH$_4^+$)")
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    hi = max(x.max(), y.max()) * 1.06
    ax.plot([0, hi], [0, hi], "k--", lw=0.8, label="equal binding")
    if split is not None:
        ax.axvline(split[0], color="tab:orange", lw=1)
        ax.axhline(split[1], color="tab:orange", lw=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("|BE NH$_4^+$| (kJ/mol)")
    ax.set_ylabel("|BE Li$^+$| (kJ/mol)")
    ax.set_title(title, fontsize=10)
    ax.text(0.97, 0.03, "below the diagonal: NH$_4^+$ binds more strongly", transform=ax.transAxes,
            ha="right", fontsize=8, color="0.3")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


ION_COLOR = {"Li": "tab:blue", "NH4": "tab:orange", "Li-NH4": "tab:purple"}


def group_effects(eff, path, title):
    """Contact-group effects relative to the reference group, with 95% intervals, one marker per ion."""
    e = eff[eff["term"] != "net charge"]
    if e.empty:
        return
    groups = sorted(e["term"].unique(), key=lambda g: e[e["term"] == g]["effect"].mean())
    ions = list(e["ion"].unique())
    fig, ax = plt.subplots(figsize=(7.5, 0.6 * len(groups) + 1.6))
    for k, ion in enumerate(ions):
        sub = e[e["ion"] == ion].set_index("term")
        off = (k - (len(ions) - 1) / 2) * 0.22
        for i, g in enumerate(groups):
            if g not in sub.index:
                continue
            r = sub.loc[g]
            ax.errorbar(r["effect"], i + off, xerr=[[max(r["effect"] - r["ci_lo"], 0)], [max(r["ci_hi"] - r["effect"], 0)]],
                        fmt="o", color=ION_COLOR.get(ion, "k"), capsize=3, label=ION_LABEL.get(ion, ion) if i == 0 or g == groups[0] else None)
            ax.annotate("%d mol" % r["n_molecules"], (r["ci_hi"], i + off), xytext=(4, -3), textcoords="offset points",
                        fontsize=7, color="0.4")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    ref = e["relative_to"].iloc[0]
    t = e["target"].iloc[0]
    ax.set_xlabel("effect on %s relative to %s, net charge held fixed (kJ/mol)" % (TARGET_LABEL.get(t, t), ref))
    ax.set_title(title, fontsize=10)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def group_distance(resid, coef, draws, path, title):
    """
    One panel per functional group: partial residual of the target against the distance from
    the ion to the nearest group of that type, with the fitted beta/d curve and its 95% band.
    """
    if resid.empty:
        return
    groups = list(dict.fromkeys(resid["group"]))
    nc = 3
    nr = int(np.ceil(len(groups) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.3 * nc, 3.4 * nr), squeeze=False)
    for ax, g in zip(axes.flat, groups):
        d = np.linspace(1.6, max(10, resid[resid["group"] == g]["distance"].max() + 0.5), 200)
        for ion in resid["ion"].unique():
            r = resid[(resid["group"] == g) & (resid["ion"] == ion)]
            if r.empty:
                continue
            col = ION_COLOR.get(ion, "k")
            pt = r["proton_transfer"] == 1
            ax.scatter(r.loc[~pt, "distance"], r.loc[~pt, "partial_residual"], s=16, color=col, alpha=0.8,
                       label=ION_LABEL.get(ion, ion))
            if pt.any():
                ax.scatter(r.loc[pt, "distance"], r.loc[pt, "partial_residual"], s=50, marker="^", color="tab:red",
                           edgecolors="k", linewidths=0.5, label="proton transfer")
            c = coef[(coef["ion"] == ion) & (coef["group"] == g)]
            if c.empty:
                continue
            c = c.iloc[0]
            ax.plot(d, c["beta_kjmol_A"] / d, color=col, lw=1.6)
            dr = draws[draws["ion"] == ion][c["column"]].dropna().values if c["column"] in draws else []
            if len(dr):
                band = np.percentile(np.outer(dr, 1 / d), [2.5, 97.5], axis=0)
                ax.fill_between(d, band[0], band[1], color=col, alpha=0.15, lw=0)
            ax.text(0.98, 0.04 + 0.09 * list(resid["ion"].unique()).index(ion),
                    "%s: %.0f/d  [%.0f, %.0f]" % (ION_LABEL.get(ion, ion), c["beta_kjmol_A"], c["ci_lo"], c["ci_hi"]),
                    transform=ax.transAxes, ha="right", fontsize=7, color=col)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(g, fontsize=9)
        ax.set_xlabel("distance from ion to nearest group (Å)", fontsize=8)
        ax.set_ylabel("partial effect (kJ/mol)", fontsize=8)
        ax.set_xlim(1.5, max(10, resid[resid["group"] == g]["distance"].max() + 0.5))
    for ax in list(axes.flat)[len(groups):]:
        ax.axis("off")
    seen = {}
    for ax in axes.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(ll, hh)
    if seen:
        fig.legend(seen.values(), seen.keys(), loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)
