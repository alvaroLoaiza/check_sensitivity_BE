"""
Export the geometry of selected complexes as small XYZ files, for figures and for
viewing with the molecule viewers (molecule_viewer.html accepts plain XYZ lines).

Each file holds the LAST orientation block of the complex single-point log (the
geometry the energies belong to), monomer atoms first and ion atoms last. The
comment line carries what the analysis knows about the row, and index.csv lists
everything, so the files can be shared without the 30 MB logs.
"""
from pathlib import Path

import pandas as pd

from .features.common import dist, parse_geometry, read_text

# Default examples: proton transfer (full, partial, none at the same group type)
# and the most NH4+-selective sites without proton transfer.
EXAMPLES = ["L1_1", "M1_2", "L2_2", "M2_1", "F3", "O3", "N2", "A2"]


def _fmt(v, f="%.1f"):
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else f % v


def write_xyz(atoms, path, comment):
    lines = [str(len(atoms)), comment.replace("\n", " ")]
    lines += ["%-2s %12.6f %12.6f %12.6f" % a for a in atoms]
    Path(path).write_text("\n".join(lines) + "\n")


def export(manifest, labels, out, features=None, ions=("Li", "NH4")):
    """
    Write <out>/<label>_<ion>.xyz for every requested label and ion that has a
    complex log, plus <out>/index.csv. Returns the index DataFrame.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    feat = features.set_index(["label", "ion"]) if features is not None else None
    rows = []
    for lab in labels:
        for ion in ions:
            m = manifest[(manifest["label"] == lab) & (manifest["ion"] == ion)]
            if m.empty or pd.isna(m.iloc[0].get("complex_log")) or not str(m.iloc[0]["complex_log"]):
                rows.append(dict(label=lab, ion=ion, file="", note="no complex log"))
                continue
            atoms = parse_geometry(read_text(m.iloc[0]["complex_log"]))
            n_ion = 1 if ion == "Li" else 5
            info = dict(label=lab, ion=ion, n_atoms=len(atoms),
                        ion_atoms="%d-%d" % (len(atoms) - n_ion + 1, len(atoms)))
            f = feat.loc[(lab, ion)] if feat is not None and (lab, ion) in feat.index else None
            if f is not None:
                info.update(be_kjmol=f.get("be_kjmol"), contact_group=f.get("contact_group"),
                            contact_dist=f.get("contact_dist"), proton_transfer=f.get("proton_transfer"),
                            ion_NH_max=f.get("ion_NH_max"), ion_charge_nbo=f.get("ion_charge_nbo"))
            if ion == "NH4" and len(atoms) >= 5:
                n = len(atoms) - 5
                info["ion_NH_max_geom"] = max(dist(atoms[n], atoms[h]) for h in range(n + 1, n + 5))
            comment = " | ".join(x for x in [
                "%s %s" % (lab, ion),
                "BE %s kJ/mol" % _fmt(info.get("be_kjmol")) if info.get("be_kjmol") is not None else "",
                "contact %s %s A" % (info.get("contact_group"), _fmt(info.get("contact_dist"), "%.2f"))
                if info.get("contact_group") else "",
                "proton transfer" if info.get("proton_transfer") == 1 else "",
                "N-H max %s A" % _fmt(info.get("ion_NH_max_geom"), "%.2f") if ion == "NH4" else "",
                "ion atoms %s" % info["ion_atoms"],
            ] if x)
            name = "%s_%s.xyz" % (lab, ion)
            write_xyz(atoms, out / name, comment)
            info.update(file=name, note="")
            rows.append(info)
    idx = pd.DataFrame(rows)
    idx.to_csv(out / "index.csv", index=False)
    return idx
