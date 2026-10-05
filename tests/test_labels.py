from pathlib import Path

import pytest

from check_sensitivity_be.be_table import load_be_table
from check_sensitivity_be.labels import (bare_dir_matches, complex_dir_name,
                                         parse_label)

DATA = Path(__file__).resolve().parents[1] / "data" / "all_BE_values.xlsx"


@pytest.mark.parametrize("raw,mol,site", [
    ("A1", "A", 1), ("C4", "C", 4), ("K5", "K", 5),
    ("L1_1", "L1", 1), ("L2_2", "L2", 2), ("M1_2", "M1", 2), ("P1_6", "P1", 6), ("Q2_3", "Q2", 3),
    ("TG", "TG", 0), ("CH3SO3", "CH3SO3", 0),
])
def test_parse_label(raw, mol, site):
    lab = parse_label(raw)
    assert (lab.mol, lab.site) == (mol, site)


def test_unknown_label_raises():
    with pytest.raises(ValueError):
        parse_label("ZZ99")


def test_every_label_in_the_real_table_parses():
    be = load_be_table(DATA)
    assert len(be) == 68 * 2           # 68 monomer sites x 2 ions
    assert be["be_kjmol"].notna().sum() == 122   # 60 NH4+ + 62 Li+


@pytest.mark.parametrize("mol,ion,expected", [
    ("C", "Li", "080.Li_mol_C"),
    ("P1", "NH4", "080.NH4_mol_P1"),
    ("A", "NH4", "070.NH4_C22H24BO3"),
    ("B", "Li", "075.Li_C11H9N2O2S"),
    ("CH3SO3", "Li", "060.Li_CH3SO3"),
    ("TG", "Li", "060.Li_p_TG_2"),       # TG_1 is the empty one
])
def test_complex_dir_name(mol, ion, expected):
    assert complex_dir_name(mol, ion) == expected


@pytest.mark.parametrize("mol,dirname,expected", [
    ("C", "050.mol_C_m", True),
    ("F", "050.mol_F_2m", True),
    ("L1", "050.mol_L1_2m", True),
    ("M1", "050.mol_M1_2m", True),
    ("M1", "050.mol_M2_m", False),      # M1 must not match M2
    ("P1", "050.mol_P1", True),
    ("P1", "050.mol_P2", False),
    ("C", "050.mol_C1_m", False),       # no digit-suffixed cousin
    ("A", "050.C22H24BO3_m", True),
    ("B", "050.C11H9N2O2S_m", True),
    ("TG", "050.TG", True),
])
def test_bare_dir_matches(mol, dirname, expected):
    assert bare_dir_matches(mol, dirname) is expected
