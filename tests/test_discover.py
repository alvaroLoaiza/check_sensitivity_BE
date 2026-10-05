import pandas as pd
import pytest

from check_sensitivity_be import cli
from check_sensitivity_be.discover import build_manifest, find_singlep_log
from check_sensitivity_be.energies import HARTREE_TO_KJMOL, read_log_info


def write_log(path, energy, charge, natoms, normal=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (" Charge = %2d Multiplicity = 1\n NAtoms=%7d NQM=%7d\n"
            " SCF Done:  E(RPBE1PBE) = %16.10f     A.U. after    1 cycles\n" % (charge, natoms, natoms, energy))
    if normal:
        text += " Normal termination of Gaussian 16 at Mon Jan  1 00:00:00 2026.\n"
    path.write_text(text)


def be_table(rows):
    return pd.DataFrame(rows, columns=["label", "mol", "site", "ion", "be_kjmol"])


@pytest.fixture
def tree(tmp_path):
    """C site 1 (Li) fully present with a known BE; TG present; M1_2 missing its site folder."""
    root = tmp_path / "LONI_work_folder"
    sp = "OPT_h/singlep"
    E_ion, E_mono, dE = -7.0, -100.0, -0.02           # complex = mono + ion + dE
    write_log(root / "010.Li_p" / sp / "Li_p_high_singlep.log", E_ion, +1, 1)
    write_log(root / "000.NH4_p" / sp / "NH4_p_high_singlep.log", -57.0, +1, 5)
    write_log(root / "050.mol_C_m" / sp / "mol_C_high_singlep.log", E_mono, -1, 20)
    write_log(root / "080.Li_mol_C/1" / sp / "C_Li_site1_high_singlep.log", E_mono + E_ion + dE, 0, 21)
    # TG: monomer 050.TG, complex only in the _TG_2/0 folder; _TG_1 exists but is empty
    write_log(root / "050.TG" / sp / "TG_high_singlep.log", -50.0, -1, 10)
    write_log(root / "060.Li_p_TG_2/0" / sp / "TG_Li_high_singlep.log", -50.0 + E_ion - 0.01, 0, 11)
    (root / "060.Li_p_TG_1" / "0").mkdir(parents=True)
    # M1: monomer and complex dir exist, but only site folder '1' (label asks for site 2)
    write_log(root / "050.mol_M1_2m" / sp / "mol_M1_high_singlep.log", -200.0, -2, 30)
    write_log(root / "080.Li_mol_M1/1" / sp / "M1_Li_high_singlep.log", -207.1, -1, 31)
    return root, E_ion, E_mono, dE


def test_manifest_categories(tree):
    root, *_ = tree
    be = be_table([
        ("C1", "C", 1, "Li", -52.5),       # all present -> usable
        ("TG", "TG", 0, "Li", -26.0),      # TG special-case folder -> usable
        ("M1_2", "M1", 2, "Li", -90.0),    # site folder missing -> be_but_logs_missing
        ("C1", "C", 1, "NH4", float("nan")),  # NH4 complex absent, no BE -> no_be_no_logs
    ])
    m = build_manifest(root, be).set_index(["label", "ion"])
    assert m.loc[("C1", "Li"), "category"] == "usable"
    assert m.loc[("TG", "Li"), "category"] == "usable"
    assert "060.Li_p_TG_2" in m.loc[("TG", "Li"), "complex_log"]
    row = m.loc[("M1_2", "Li")]
    assert row["category"] == "be_but_logs_missing"
    assert row["complex_status"] == "site_dir_not_found"
    assert "1" in row["complex_detail"]            # tells you which site folders DO exist
    assert m.loc[("C1", "NH4"), "category"] == "no_be_no_logs"


def test_ambiguous_logs_are_flagged(tmp_path):
    d = tmp_path / "x" / "OPT_h" / "singlep"
    d.mkdir(parents=True)
    (d / "a.log").write_text("")
    (d / "b.log").write_text("")
    path, status, detail = find_singlep_log(tmp_path / "x")
    assert path is None and status == "ambiguous_logs"
    (d / "b_singlep.log").write_text("")
    # exactly one *_singlep.log among several -> preferred
    (d / "a.log").unlink()
    assert find_singlep_log(tmp_path / "x")[1] == "ok"


def test_read_log_info_takes_last_scf_and_first_charge(tmp_path):
    p = tmp_path / "t.log"
    p.write_text(" Charge = -1 Multiplicity = 1\n NAtoms=  3 NQM= 3\n"
                 " SCF Done:  E(RPBE1PBE) =  -1.5  A.U.\n SCF Done:  E(RPBE1PBE) =  -2.5  A.U.\n"
                 " Charge =  7 Multiplicity = 1\n Normal termination of Gaussian\n")
    info = read_log_info(p)
    assert (info.energy_ha, info.charge, info.natoms, info.normal_termination) == (-2.5, -1, 3, True)


def test_verify_recomputes_be_and_flags_a_wrong_row(tree, tmp_path, capsys):
    root, E_ion, E_mono, dE = tree
    be = be_table([
        ("C1", "C", 1, "Li", dE * HARTREE_TO_KJMOL),   # table value agrees with the logs
        ("TG", "TG", 0, "Li", -999.0),                 # table value disagrees -> flagged
    ])
    mpath, vpath = tmp_path / "manifest.csv", tmp_path / "verify.csv"
    build_manifest(root, be).to_csv(mpath, index=False)
    cli.main(["verify", "--manifest", str(mpath), "-o", str(vpath)])
    res = pd.read_csv(vpath).set_index("label")
    assert abs(res.loc["C1", "d_be"]) < 1e-6
    assert res.loc["C1", "charge_ok"] and res.loc["C1", "natoms_ok"]
    assert abs(res.loc["TG", "d_be"]) > 100
    out = capsys.readouterr().out
    assert "1 of 2" in out                              # only one BE matched
