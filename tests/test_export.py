"""export: XYZ of selected complexes, annotated from features.csv."""
from pathlib import Path

import pandas as pd

from check_sensitivity_be import cli
from check_sensitivity_be.export import export

ORIENT = """                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
%s ---------------------------------------------------------------------
"""


def _log(path, atoms):
    rows = "".join("%7d%11d           0   %12.6f%12.6f%12.6f\n" % (i + 1, z, x, y, zz)
                   for i, (z, x, y, zz) in enumerate(atoms))
    path.write_text(ORIENT % (" 1 1 0 9.0 9.0 9.0\n") + ORIENT % rows + " Normal termination of Gaussian\n")


def test_export_writes_last_geometry_and_annotation(tmp_path):
    li, nh = tmp_path / "li.log", tmp_path / "nh.log"
    _log(li, [(8, 0, 0, 0), (3, 1.9, 0, 0)])
    # O, then NH4+ with one H moved 1.0 A from O (1.76 A from N)
    _log(nh, [(8, 0, 0, 0), (7, 2.76, 0, 0), (1, 1.0, 0, 0), (1, 3.1, 0.9, 0), (1, 3.1, -0.5, 0.8), (1, 3.1, -0.5, -0.8)])
    man = pd.DataFrame(dict(label=["X1", "X1"], ion=["Li", "NH4"], complex_log=[str(li), str(nh)]))
    feats = pd.DataFrame(dict(label=["X1", "X1"], ion=["Li", "NH4"], be_kjmol=[-40.0, -90.0],
                              contact_group=["ether(C-O-C)"] * 2, contact_dist=[1.9, 2.76], proton_transfer=[0, 1]))
    idx = export(man, ["X1", "Y9"], tmp_path / "ex", features=feats)
    lines = (tmp_path / "ex" / "X1_NH4.xyz").read_text().splitlines()
    assert lines[0] == "6" and "proton transfer" in lines[1] and "N-H max 1.76 A" in lines[1]
    assert lines[1].endswith("ion atoms 2-6")
    assert lines[2].split()[0] == "O"            # last orientation block, not the input echo
    li_lines = (tmp_path / "ex" / "X1_Li.xyz").read_text().splitlines()
    assert li_lines[0] == "2" and "BE -40.0 kJ/mol" in li_lines[1]
    assert set(idx[idx.label == "Y9"].note) == {"no complex log"}


def test_export_command(tmp_path):
    log = tmp_path / "li.log"
    _log(log, [(8, 0, 0, 0), (3, 1.9, 0, 0)])
    pd.DataFrame(dict(label=["X1"], ion=["Li"], complex_log=[str(log)], category=["usable"])
                 ).to_csv(tmp_path / "manifest.csv", index=False)
    cli.main(["export", "--labels", "X1", "--manifest", str(tmp_path / "manifest.csv"),
              "--features", str(tmp_path / "none.csv"), "-o", str(tmp_path / "ex")])
    assert (tmp_path / "ex" / "X1_Li.xyz").is_file() and (tmp_path / "ex" / "index.csv").is_file()
