"""PNG plots for `analyze --plots` (matplotlib, no display needed)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ION_LABEL = {"Li": "Li$^+$", "NH4": "NH$_4^+$"}


def correlation_forest(corr, path, top=25):
    """rho with 95% CI per feature, one panel per ion, strongest |rho| first."""
    ions = sorted(corr["ion"].unique())
    fig, axes = plt.subplots(1, len(ions), figsize=(6.5 * len(ions), 0.32 * top + 1.5), squeeze=False)
    for ax, ion in zip(axes[0], ions):
        c = corr[corr["ion"] == ion].dropna(subset=["rho"])
        c = c.reindex(c["rho"].abs().sort_values(ascending=False).index).head(top).iloc[::-1]
        y = np.arange(len(c))
        # a percentile bootstrap CI can sit slightly off the point estimate; clip for drawing
        err = np.clip(np.vstack([c["rho"] - c["ci_lo"], c["ci_hi"] - c["rho"]]), 0, None)
        colors = ["tab:blue" if e else "0.6" for e in c["ci_excludes_0"]]
        ax.errorbar(c["rho"], y, xerr=err, fmt="none", ecolor="0.5", elinewidth=1, capsize=2)
        ax.scatter(c["rho"], y, c=colors, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(c["feature"], fontsize=8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("Spearman rho with BE (95% CI, molecule bootstrap)")
        ax.set_title("%s  (grey: CI includes 0)" % ION_LABEL.get(ion, ion))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parity(preds, metrics, feature_set, path):
    """Held-out (leave-one-molecule-out) predicted vs observed BE; proton-transfer rows marked."""
    p = preds[preds["model"] == feature_set]
    ions = sorted(p["ion"].unique())
    fig, axes = plt.subplots(1, len(ions), figsize=(5 * len(ions), 4.8), squeeze=False)
    for ax, ion in zip(axes[0], ions):
        g = p[p["ion"] == ion]
        pt = g["proton_transfer"] == 1
        ax.scatter(g.loc[~pt, "be_kjmol"], g.loc[~pt, "be_pred"], s=22, label="sites")
        if pt.any():
            ax.scatter(g.loc[pt, "be_kjmol"], g.loc[pt, "be_pred"], s=60, marker="^",
                       color="tab:red", label="proton transfer")
            for _, r in g[pt].iterrows():
                ax.annotate(r["label"], (r["be_kjmol"], r["be_pred"]), fontsize=7,
                            xytext=(4, -10), textcoords="offset points")
        lo = min(g["be_kjmol"].min(), g["be_pred"].min()) - 5
        hi = max(g["be_kjmol"].max(), g["be_pred"].max()) + 5
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        m = metrics[(metrics["ion"] == ion) & (metrics["model"] == feature_set)].iloc[0]
        ax.set_title("%s  R$^2$ = %.2f, MAE = %.1f kJ/mol" % (ION_LABEL.get(ion, ion), m["R2"], m["MAE"]))
        ax.set_xlabel("BE from DFT (kJ/mol)")
        ax.set_ylabel("BE predicted, molecule held out (kJ/mol)")
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(feature_set)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def families(fam, order, path):
    """Per family: held-out R2 alone, and R2 lost when removed from the full model."""
    ions = sorted(fam["ion"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, col, title in ((axes[0], "R2_alone", "R$^2$, family alone"),
                           (axes[1], "R2_lost_vs_full", "R$^2$ lost, family removed from full model")):
        fams = [f for f in order if f in set(fam["family"])]
        w = 0.8 / len(ions)
        for k, ion in enumerate(ions):
            v = fam[fam["ion"] == ion].set_index("family").reindex(fams)[col]
            ax.barh(np.arange(len(fams)) + (k - (len(ions) - 1) / 2) * w, v, height=w,
                    label=ION_LABEL.get(ion, ion))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(fams)))
        ax.set_yticklabels(fams)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("leave-one-molecule-out")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def family_shap_strip(shap_fam, order, path):
    """Each site's summed SHAP per family; proton-transfer rows as red triangles."""
    ions = sorted(shap_fam["ion"].unique())
    fig, axes = plt.subplots(1, len(ions), figsize=(6.5 * len(ions), 4.2), squeeze=False)
    fig.subplots_adjust(wspace=0.5)
    rng = np.random.default_rng(0)
    for ax, ion in zip(axes[0], ions):
        s = shap_fam[shap_fam["ion"] == ion]
        fams = [f for f in order if f in s.columns and s[f].notna().any()]
        for k, f in enumerate(fams):
            pt = s["proton_transfer"] == 1
            jit = rng.uniform(-0.18, 0.18, len(s))
            ax.scatter(s.loc[~pt, f], (k + jit)[~pt.values], s=14, color="tab:blue")
            ax.scatter(s.loc[pt, f], (k + jit)[pt.values], s=40, marker="^", color="tab:red")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels(fams)
        ax.invert_yaxis()
        ax.set_xlabel("summed SHAP of the family (kJ/mol; negative = stronger binding)")
        ax.set_title("%s  (red: proton transfer)" % ION_LABEL.get(ion, ion))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
