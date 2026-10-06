"""
One feature row per (site, ion), from three single-point logs:

    complex  -> NBO charges and geometry around the ion (monomer atoms only)
    monomer  -> HOMO energy, HOMO share on the contact group (pop=full)
    ion      -> LUMO energy, used only through the cross-fragment gap

    gap_cross_eV = LUMO(ion) - HOMO(monomer)

The bare ion's LUMO is one constant per ion, so it is not written as its own
column (it would just restate the ion). Each stage that fails leaves its
columns empty and adds a note to `errors`, so a bad file shows up as a gap in
the row, not as a silently missing row.
"""
import re
from dataclasses import dataclass, field
from typing import Optional

from ..energies import HARTREE_TO_KJMOL, _SCF
from .charges import compute_descriptors, get_charges
from .common import dist, parse_geometry, probe_index, read_text, split_fragments
from .groups import CANONICAL_GROUP_TYPES, classify_groups, group_distances, safe_name, summary_by_type
from .orbitals import HARTREE_TO_EV, homo_lumo, parse_mo_composition, parse_orbital_energies, site_fraction

DEFAULT_RADII = (3.0, 4.0, 5.0)


_CHARGE_ML = re.compile(r"^\s*Charge\s*=\s*(-?\d+)\s+Multiplicity", re.M)


def net_charge(text):
    m = _CHARGE_ML.search(text)
    return int(m.group(1)) if m else None


def last_scf_energy(text):
    found = _SCF.findall(text)
    return float(found[-1]) if found else None


@dataclass
class Fragment:
    """What the feature stage needs from a bare monomer or bare ion log."""
    elems: list
    energy_ha: Optional[float]
    homo_ha: Optional[float]
    lumo_ha: Optional[float]
    charge: Optional[int] = None
    occ: list = field(default_factory=list)        # occupied eigenvalues, Hartree
    virt: list = field(default_factory=list)       # virtual eigenvalues, Hartree
    mos: dict = field(default_factory=dict)        # {"HOMO-2".."LUMO+2": coeffs data} (monomers only)


def read_fragment(path, want_homo_coeffs):
    text = read_text(path)
    occ, virt = parse_orbital_energies(text)
    homo, lumo = homo_lumo(occ, virt)
    mos = {}
    if want_homo_coeffs and occ:
        numbers = {mo: lab for mo, lab in frontier_mo_numbers(len(occ)).items() if mo >= 1}
        parsed = parse_mo_composition(text, list(numbers))
        mos = {lab: parsed[mo] for mo, lab in numbers.items() if parsed.get(mo, {}).get("coeffs_by_atom")}
    return Fragment([a[0] for a in parse_geometry(text)], last_scf_energy(text), homo, lumo,
                    net_charge(text), occ, virt, mos)


FRONTIER = ["HOMO-2", "HOMO-1", "HOMO", "LUMO", "LUMO+1", "LUMO+2"]
_TAG = {"HOMO-2": "homo_m2", "HOMO-1": "homo_m1", "HOMO": "homo",
        "LUMO": "lumo", "LUMO+1": "lumo_p1", "LUMO+2": "lumo_p2"}


def frontier_mo_numbers(n_occ):
    """1-based MO numbers of HOMO-2 .. LUMO+2."""
    return {n_occ - 2: "HOMO-2", n_occ - 1: "HOMO-1", n_occ: "HOMO",
            n_occ + 1: "LUMO", n_occ + 2: "LUMO+1", n_occ + 3: "LUMO+2"}


def frontier_energies(occ, virt):
    """{"HOMO-2": Ha, ...} for the orbitals that exist."""
    out = {}
    for k, lab in enumerate(["HOMO", "HOMO-1", "HOMO-2"]):
        if len(occ) > k:
            out[lab] = occ[-1 - k]
    for k, lab in enumerate(["LUMO", "LUMO+1", "LUMO+2"]):
        if len(virt) > k:
            out[lab] = virt[k]
    return out


class FragmentCache:
    """Each bare monomer / ion log is read once per run, however many sites share it."""

    def __init__(self):
        self._d = {}

    def get(self, path, want_homo_coeffs=False):
        key = (str(path), want_homo_coeffs)
        if key not in self._d:
            self._d[key] = read_fragment(path, want_homo_coeffs)
        return self._d[key]


def feature_columns(radii=DEFAULT_RADII):
    """Fixed column order, so every row has the same columns whatever was detected."""
    cols = ["be_kjmol", "ion_charge_nbo", "proton_transfer", "ion_NH_max", "charge_source"]
    for r in radii:
        t = "%g" % r
        cols += ["V_r" + t, "Efield_r" + t, "Qnet_r" + t, "n_env_r" + t]
    cols += ["V_all", "Efield_all"]
    cols += ["nearest_atom_elem", "nearest_atom_dist", "nearest_atom_charge",
             "contact_group", "contact_dist"]
    for ct in CANONICAL_GROUP_TYPES:
        cols += ["count_" + safe_name(ct), "dist_" + safe_name(ct)]
    cols += ["homo_mon_eV", "homo_mon_frac_contact", "gap_cross_eV"]
    cols += ["mon_%s_eV" % _TAG[l] for l in FRONTIER if l != "HOMO"]
    cols += ["mon_gap0_eV", "mon_gap1_eV", "mon_gap2_eV"]
    cols += ["mon_frac_contact_%s" % _TAG[l] for l in FRONTIER if l != "HOMO"]
    cols += ["errors"]
    return cols


def proton_transfer_check(atoms, p, ion_idx, mon_idx):
    """
    ion_NH_max: longest N-H inside the ion fragment (about 1.03 Angstrom for intact NH4+).
    proton_transfer: 1 if any ion H is closer to a monomer N/O than to the ion's N.
    For Li+ (no H): 0 and None.
    """
    hs = [i for i in ion_idx if atoms[i][0] == "H"]
    if not hs:
        return {"proton_transfer": 0, "ion_NH_max": None}
    acceptors = [i for i in mon_idx if atoms[i][0] in ("N", "O")]
    moved = 0
    for h in hs:
        if acceptors and min(dist(atoms[h], atoms[a]) for a in acceptors) < dist(atoms[h], atoms[p]):
            moved = 1
    return {"proton_transfer": moved, "ion_NH_max": max(dist(atoms[h], atoms[p]) for h in hs)}


def extract_row(complex_path, monomer_path, ion_path, cache, radii=DEFAULT_RADII):
    row = {c: None for c in feature_columns(radii)}
    errors = []

    mono = cache.get(monomer_path, want_homo_coeffs=True)
    ion = cache.get(ion_path)
    text = read_text(complex_path)
    atoms = parse_geometry(text)

    e_c = last_scf_energy(text)
    if None not in (e_c, mono.energy_ha, ion.energy_ha):
        row["be_kjmol"] = (e_c - mono.energy_ha - ion.energy_ha) * HARTREE_TO_KJMOL
    else:
        errors.append("missing SCF energy")

    try:
        mon_idx, ion_idx = split_fragments(atoms, mono.elems, ion.elems)
        p = probe_index(atoms, ion_idx)
    except ValueError as e:
        errors.append("fragments: %s" % e)
        row["errors"] = "; ".join(errors)
        return row

    # --- did an NH4+ proton move onto the monomer? (geometry only) ---
    row.update(proton_transfer_check(atoms, p, ion_idx, mon_idx))

    # --- charges (complex) ---
    charges, source = get_charges(text)
    if not charges:
        errors.append("no charge table")
    elif len(charges) != len(atoms):
        errors.append("charge table has %d entries for %d atoms" % (len(charges), len(atoms)))
    else:
        row["charge_source"] = source
        row["ion_charge_nbo"] = sum(charges[i] for i in ion_idx)
        row.update(compute_descriptors(atoms, charges, p, radii, mon_idx))

    # --- functional groups (complex geometry, monomer atoms only) ---
    pairs = []
    try:
        groups = classify_groups(atoms[:len(mon_idx)])
        pairs = group_distances(groups, atoms, atoms[p][1:4])
        by_type = summary_by_type(pairs)
        for ct in CANONICAL_GROUP_TYPES:
            count, d = by_type.get(ct, (0, None))
            row["count_" + safe_name(ct)] = count
            row["dist_" + safe_name(ct)] = d
        if pairs:
            row["contact_group"] = pairs[0]["type"]
            row["contact_dist"] = pairs[0]["dist"]
        else:
            errors.append("no functional group detected")
    except Exception as e:  # keep the row; the error column says what broke
        errors.append("groups: %s" % e)

    # --- orbitals (bare fragments) ---
    if mono.homo_ha is None or ion.lumo_ha is None:
        errors.append("missing orbital eigenvalues")
    else:
        row["homo_mon_eV"] = mono.homo_ha * HARTREE_TO_EV
        row["gap_cross_eV"] = (ion.lumo_ha - mono.homo_ha) * HARTREE_TO_EV
    e = frontier_energies(mono.occ, mono.virt)
    for lab in FRONTIER:
        if lab != "HOMO" and lab in e:
            row["mon_%s_eV" % _TAG[lab]] = e[lab] * HARTREE_TO_EV
    for k, (h, l) in enumerate([("HOMO", "LUMO"), ("HOMO-1", "LUMO+1"), ("HOMO-2", "LUMO+2")]):
        if h in e and l in e:
            row["mon_gap%d_eV" % k] = (e[l] - e[h]) * HARTREE_TO_EV
    if pairs:
        if mono.mos.get("HOMO"):
            site = pairs[0]["atoms"]
            for lab, data in mono.mos.items():
                frac = site_fraction(data, site)
                if lab == "HOMO":
                    row["homo_mon_frac_contact"] = frac
                else:
                    row["mon_frac_contact_%s" % _TAG[lab]] = frac
        else:
            errors.append("no MO coefficients in monomer log (needs pop=full)")

    row["errors"] = "; ".join(errors) if errors else ""
    return row
