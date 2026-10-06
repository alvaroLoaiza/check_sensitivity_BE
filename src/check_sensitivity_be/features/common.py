"""
Shared geometry parsing and bond-graph helpers (port of g16features_common.py).

The standalone scripts auto-detected the ion inside one log. The package does
not need to: the bare monomer and bare ion logs tell us how many atoms each
fragment has, and in every complex the monomer atoms come first and the ion
atoms last. `split_fragments` uses that and checks it against the element
sequence, so a wrong split fails loudly instead of producing wrong features.
"""
import math
import re

Z2SYM = {1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
         11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K",
         20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
         29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 47: "Ag",
         53: "I", 78: "Pt", 79: "Au"}

COVALENT_RADIUS = {
    "H": 0.31, "HE": 0.28, "LI": 1.28, "BE": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "NE": 0.58, "NA": 1.66, "MG": 1.41, "AL": 1.21, "SI": 1.11, "P": 1.07, "S": 1.05,
    "CL": 1.02, "AR": 1.06, "K": 2.03, "CA": 1.76, "TI": 1.60, "FE": 1.32, "ZN": 1.22,
    "BR": 1.20, "AG": 1.45, "I": 1.39, "PT": 1.36, "AU": 1.36,
}
DEFAULT_RADIUS = 0.75


def radius_for(elem):
    return COVALENT_RADIUS.get(elem.upper(), DEFAULT_RADIUS)


_ORIENT = re.compile(
    r"(Standard orientation|Input orientation|Z-Matrix orientation):.*?\n.*?\n.*?\n.*?\n.*?\n(.*?)\n\s*-{5,}",
    re.DOTALL)


def parse_geometry(text):
    """[(elem, x, y, z), ...] from the LAST orientation block."""
    blocks = list(_ORIENT.finditer(text))
    if not blocks:
        return []
    atoms = []
    for line in blocks[-1].group(2).splitlines():
        tokens = line.split()
        if len(tokens) != 6:
            continue
        try:
            z = int(tokens[1])
            x, y, zz = float(tokens[3]), float(tokens[4]), float(tokens[5])
        except ValueError:
            continue
        elem = Z2SYM.get(z)
        if elem:
            atoms.append((elem, x, y, zz))
    return atoms


def read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def dist(a, b):
    return math.sqrt((a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 + (a[3] - b[3]) ** 2)


def build_bonds(atoms, tolerance=1.15):
    """(i, j, d) for every pair within (covalent radius sum) * tolerance."""
    bonds = []
    n = len(atoms)
    for i in range(n):
        ri = radius_for(atoms[i][0])
        for j in range(i + 1, n):
            d = dist(atoms[i], atoms[j])
            if d <= (ri + radius_for(atoms[j][0])) * tolerance:
                bonds.append((i, j, d))
    return bonds


def adjacency(atoms, bonds):
    adj = {i: [] for i in range(len(atoms))}
    for i, j, d in bonds:
        adj[i].append((j, d))
        adj[j].append((i, d))
    return adj


def split_fragments(complex_atoms, monomer_elems, ion_elems):
    """
    Returns (monomer_indices, ion_indices) into the complex, assuming the
    monomer atoms come first and the ion atoms last. Raises ValueError if the
    element sequences do not match the bare-fragment logs.
    """
    n_m, n_i = len(monomer_elems), len(ion_elems)
    elems = [a[0] for a in complex_atoms]
    if len(elems) != n_m + n_i:
        raise ValueError("complex has %d atoms, monomer + ion = %d" % (len(elems), n_m + n_i))
    if elems[:n_m] != list(monomer_elems):
        raise ValueError("first %d complex atoms do not match the monomer's element order" % n_m)
    if elems[n_m:] != list(ion_elems):
        raise ValueError("last %d complex atoms %s do not match the ion %s"
                         % (n_i, elems[n_m:], list(ion_elems)))
    return list(range(n_m)), list(range(n_m, n_m + n_i))


def probe_index(complex_atoms, ion_indices):
    """The ion's centre atom: Li for Li+, N for NH4+ (the only heavy atom of the ion)."""
    heavy = [i for i in ion_indices if complex_atoms[i][0] != "H"]
    if len(heavy) != 1:
        raise ValueError("ion fragment has %d heavy atoms, expected 1" % len(heavy))
    return heavy[0]
