"""
Regression tests on real logs (A1, Li+). The logs are gitignored, so these skip
unless sample_logs/ holds:
    Li_site1_high_singlep.log, C22H24BO3_m_high_singlep.log, Li_p_high_singlep.log
Reference values come from the standalone g16extract_all.py (9/30/2026) on the
same files, except where the port deliberately changed the definition.
"""
from pathlib import Path

import pytest

from check_sensitivity_be.energies import read_log_info
from check_sensitivity_be.features.extract import FragmentCache, extract_row

S = Path(__file__).resolve().parents[1] / "sample_logs"
CX, MON, ION = S / "Li_site1_high_singlep.log", S / "C22H24BO3_m_high_singlep.log", S / "Li_p_high_singlep.log"
pytestmark = pytest.mark.skipif(not all(p.is_file() for p in (CX, MON, ION)), reason="sample logs not present")


@pytest.fixture(scope="module")
def row():
    return extract_row(CX, MON, ION, FragmentCache())


def test_be_matches_verify(row):
    assert row["be_kjmol"] == pytest.approx(-28.7036315, abs=1e-5)


def test_charge_descriptors_unchanged_from_standalone_script(row):
    # Li+ is one atom, so the monomer-only shell gives the script's numbers exactly
    expected = {"V_r3": 0.034658, "Efield_r3": 0.080663, "Qnet_r3": 0.42024, "n_env_r3": 7,
                "V_r4": -0.169585, "Qnet_r4": -0.27294, "n_env_r4": 15,
                "V_r5": -0.423243, "Qnet_r5": -1.31066, "n_env_r5": 30,
                "ion_charge_nbo": 0.97741, "nearest_atom_dist": 1.953053, "nearest_atom_charge": -0.75975}
    for k, v in expected.items():
        assert row[k] == pytest.approx(v, abs=2e-6), k
    assert row["nearest_atom_elem"] == "O" and row["charge_source"] == "NBO"


def test_groups(row):
    assert row["count_aromatic_ring"] == 3 and row["count_borate_boronic_BOx"] == 1
    assert row["dist_aromatic_ring"] == pytest.approx(2.872379, abs=2e-6)   # unchanged
    # changed on purpose: borate measured to its nearest O (the script used B, 2.938788)
    assert row["contact_group"] == "borate/boronic(BOx)"
    assert row["contact_dist"] == pytest.approx(1.953053, abs=2e-6)


def test_fragment_orbitals(row):
    # monomer HOMO -0.21189 Ha, Li+ LUMO 0.02489 Ha (script values on the bare logs)
    assert row["homo_mon_eV"] == pytest.approx(-0.21189 * 27.211386)
    assert row["gap_cross_eV"] == pytest.approx((0.02489 + 0.21189) * 27.211386)
    assert 0.0 < row["homo_mon_frac_contact"] < 1.0
    assert row["errors"] == ""


def test_atom_order_monomer_first_ion_last():
    c, m, i = read_log_info(CX), read_log_info(MON), read_log_info(ION)
    assert c.atomic_numbers == m.atomic_numbers + i.atomic_numbers
    assert i.atomic_numbers == (3,)


def test_monomer_frontier_orbitals(row):
    # A1 bare monomer eigenvalues (Ha): HOMO-2 -0.22067, HOMO-1 -0.21463, HOMO -0.21189,
    # LUMO 0.00072, LUMO+1 0.00173, LUMO+2 0.00244 (standalone g16orbitals, same file)
    ev = 27.211386
    assert row["mon_homo_m2_eV"] == pytest.approx(-0.22067 * ev)
    assert row["mon_lumo_p2_eV"] == pytest.approx(0.00244 * ev)
    assert row["mon_gap0_eV"] == pytest.approx((0.00072 + 0.21189) * ev)
    assert row["mon_gap1_eV"] == pytest.approx((0.00173 + 0.21463) * ev)
    assert row["mon_gap2_eV"] == pytest.approx((0.00244 + 0.22067) * ev)
    for tag in ("homo_m2", "homo_m1", "lumo", "lumo_p1", "lumo_p2"):
        assert 0.0 <= row["mon_frac_contact_" + tag] <= 1.0
