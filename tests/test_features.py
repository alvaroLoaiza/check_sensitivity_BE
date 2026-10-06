"""Unit tests for the feature stage on small synthetic inputs (no real logs needed)."""
import math

import pandas as pd
import pytest

from check_sensitivity_be import cli
from check_sensitivity_be.energies import geometry_status, opt_logs_for, read_log_info
from check_sensitivity_be.exclusions import load_exclusions
from check_sensitivity_be.features.charges import compute_descriptors, parse_nbo_charges
from check_sensitivity_be.features.common import probe_index, split_fragments
from check_sensitivity_be.features.groups import (classify_groups, group_distances,
                                                   safe_name, summary_by_type)
from check_sensitivity_be.features.orbitals import parse_orbital_energies, site_fraction

NBO = """
 Summary of Natural Population Analysis:

                                     Natural Population
                 Natural  -----------------------------------------------
    Atom  No    Charge         Core      Valence    Rydberg      Total
 -----------------------------------------------------------------------
      N    1   -0.83911      1.99966     5.82620    0.01325     7.83911
      H    2    0.45986      0.00000     0.53968    0.00047     0.54014
 =======================================================================
"""


def test_nbo_table_closed_by_equals_line():
    assert parse_nbo_charges(NBO) == {0: -0.83911, 1: 0.45986}


def _nh4_near_oxygen():
    # monomer: one O at index 0;  ion: N + 4 H, listed last
    b = 1.03 / math.sqrt(3)
    atoms = [("O", 2.8, 0.0, 0.0), ("N", 0.0, 0.0, 0.0),
             ("H", b, b, b), ("H", -b, -b, b), ("H", -b, b, -b), ("H", b, -b, -b)]
    charges = {0: -0.8, 1: -0.84, 2: 0.46, 3: 0.46, 4: 0.46, 5: 0.46}
    return atoms, charges


def test_charge_shell_ignores_the_ions_own_hydrogens():
    atoms, q = _nh4_near_oxygen()
    mon, ion = split_fragments(atoms, ["O"], ["N", "H", "H", "H", "H"])
    p = probe_index(atoms, ion)
    d = compute_descriptors(atoms, q, p, [3.0], mon)
    assert p == 1
    assert d["n_env_r3"] == 1 and d["nearest_atom_elem"] == "O"
    assert d["V_r3"] == pytest.approx(-0.8 / 2.8)


def test_split_fragments_rejects_wrong_order():
    atoms, _ = _nh4_near_oxygen()
    with pytest.raises(ValueError):
        split_fragments(atoms, ["N", "H", "H", "H", "H"], ["O"])


def _benzene(dx=0.0):
    out = []
    for k in range(6):
        a = math.pi / 3 * k
        out.append(("C", dx + 1.39 * math.cos(a), 1.39 * math.sin(a), 0.0))
    for k in range(6):
        a = math.pi / 3 * k
        out.append(("H", dx + 2.47 * math.cos(a), 2.47 * math.sin(a), 0.0))
    return out


def test_every_separate_ring_is_found():
    groups = classify_groups(_benzene() + _benzene(dx=10.0))
    assert sum(g["label"] == "aromatic_ring" for g in groups) == 2


def test_sulfonate_distance_is_to_the_nearest_oxygen_not_sulfur():
    atoms = [("C", -1.8, 0, 0), ("S", 0, 0, 0),
             ("O", 0.48, 1.37, 0), ("O", 0.48, -0.69, 1.19), ("O", 0.48, -0.69, -1.19)]
    groups = classify_groups(atoms)
    assert [g["label"] for g in groups] == ["sulfonate(SO3)"]
    probe = (0.48, 3.27, 0.0)                   # 1.90 A from the first O, ~3.3 A from S
    pair = group_distances(groups, atoms, probe)[0]
    assert pair["dist"] == pytest.approx(1.90) and pair["contact_atom"] == 2
    assert summary_by_type([pair]) == {"sulfonate(SO3)": (1, pair["dist"])}
    assert safe_name("borate/boronic(BOx)") == "borate_boronic_BOx"


def test_coordinated_carbonyl_stays_a_carbonyl():
    # classification runs on the monomer atoms only, so the ion cannot add a neighbour
    atoms = [("C", 0, 0, 0), ("O", 1.21, 0, 0), ("H", -0.55, 0.94, 0), ("H", -0.55, -0.94, 0)]
    assert [g["label"] for g in classify_groups(atoms)] == ["carbonyl(C=O)"]


def test_orbital_energies_take_last_block_and_split_run_together_numbers():
    text = (" Alpha  occ. eigenvalues --   -9.0 -1.0\n Alpha virt. eigenvalues --    0.5\n\n"
            " Alpha  occ. eigenvalues --  -10.12345-10.12340 -0.21189\n"
            " Alpha virt. eigenvalues --    0.00072   0.00173\n\n")
    occ, virt = parse_orbital_energies(text)
    assert occ == [-10.12345, -10.1234, -0.21189] and virt == [0.00072, 0.00173]


def test_site_fraction():
    mo = {"coeffs_by_atom": {0: [0.6], 1: [0.8]}}
    assert site_fraction(mo, [1]) == pytest.approx(0.64)


ORIENT = """                         Standard orientation:
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          8           0        0.000000    0.000000    0.000000
      2          3           0        1.900000    0.000000    0.000000
 ---------------------------------------------------------------------
       nuclear repulsion energy        12.3456789012 Hartrees.
"""


def test_read_log_info_gets_atom_order_and_nuclear_repulsion(tmp_path):
    p = tmp_path / "t.log"
    p.write_text(" Charge =  0 Multiplicity = 1\n" + ORIENT + " Normal termination of Gaussian\n")
    info = read_log_info(p)
    assert info.atomic_numbers == (8, 3) and info.nre_ha == pytest.approx(12.3456789012)


def test_geometry_status():
    assert geometry_status(3904.0777, [3904.0777]) == "ok"
    assert geometry_status(3906.2978, [3904.0777]) == "mismatch"
    assert geometry_status(3906.2978, [3904.0777, 3906.2978]) == "ok"
    assert geometry_status(1.0, []) == "no_opt_log"


def test_opt_logs_are_the_logs_one_level_up(tmp_path):
    sp = tmp_path / "OPT_h" / "singlep"
    sp.mkdir(parents=True)
    (tmp_path / "OPT_h" / "x_high.log").write_text("")
    (tmp_path / "OPT_h" / "RUN1").mkdir()
    (tmp_path / "OPT_h" / "RUN1" / "old.log").write_text("")      # not an opt log of this singlep
    assert [p.name for p in opt_logs_for(sp / "x_high_singlep.log")] == ["x_high.log"]


def test_exclusions(tmp_path):
    p = tmp_path / "ex.csv"
    p.write_text('label,ion,reason\nQ1_1,NH4,"stale, geometry"\n')
    assert load_exclusions(p) == {("Q1_1", "NH4"): "stale, geometry"}
    assert load_exclusions(tmp_path / "missing.csv") == {}


def test_verify_flags_singlep_on_wrong_geometry(tmp_path, capsys):
    def log(path, energy, charge, znums, nre):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join("%7d%11d           0        %d.000000    0.000000    0.000000\n" % (i + 1, z, i)
                       for i, z in enumerate(znums))
        path.write_text(" Charge = %2d Multiplicity = 1\n NAtoms=%7d\n" % (charge, len(znums))
                        + ORIENT.split("\n")[0] + "\n" + "\n".join(ORIENT.split("\n")[1:5]) + "\n"
                        + rows + " ---------------------------------------------------------------------\n"
                        + "       nuclear repulsion energy %20.10f Hartrees.\n" % nre
                        + " SCF Done:  E(RPBE1PBE) = %16.10f     A.U. after    1 cycles\n" % energy
                        + " Normal termination of Gaussian\n")

    root = tmp_path
    log(root / "ion/OPT_h/singlep/i_singlep.log", -7.0, 1, [3], 0.0)
    log(root / "ion/OPT_h/i.log", -7.0, 1, [3], 0.0)
    log(root / "mon/OPT_h/singlep/m_singlep.log", -100.0, -1, [8, 6], 20.0)
    log(root / "mon/OPT_h/m.log", -100.0, -1, [8, 6], 20.0)
    log(root / "cx/OPT_h/singlep/c_singlep.log", -107.01, 0, [8, 6, 3], 31.0)
    log(root / "cx/OPT_h/c.log", -107.0105, 0, [8, 6, 3], 30.0)          # opt ended elsewhere
    pd.DataFrame([dict(label="Z1", mol="Z", site=1, ion="Li", be_kjmol=float("nan"), has_be=False,
                       category="usable", complex_log=str(root / "cx/OPT_h/singlep/c_singlep.log"),
                       monomer_log=str(root / "mon/OPT_h/singlep/m_singlep.log"),
                       ion_log=str(root / "ion/OPT_h/singlep/i_singlep.log"))]
                 ).to_csv(root / "manifest.csv", index=False)
    cli.main(["verify", "--manifest", str(root / "manifest.csv"), "-o", str(root / "v.csv"),
              "--exclusions", str(root / "none.csv")])
    res = pd.read_csv(root / "v.csv").iloc[0]
    assert res["order_ok"] and res["geom_status"] == "mismatch" and res["geom_detail"] == "complex:mismatch"
    assert "1 mismatch" in capsys.readouterr().out
