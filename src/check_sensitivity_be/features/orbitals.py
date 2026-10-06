"""
Frontier orbital energies and site-projected MO composition (port of g16orbitals.py).

Energies come from the 'Alpha occ./virt. eigenvalues' lines (LAST block).
Composition needs pop=full ('Molecular Orbital Coefficients:'); it is a raw
sum-of-squared-coefficients fraction, NOT a Mulliken population (that needs the
AO overlap matrix, which Gaussian does not print). Use it as a relative
comparison across sites of the same molecule and basis, not an absolute number.

In the package these run on the BARE fragment logs (monomer HOMO, ion LUMO),
not on the complex.
"""
import re

HARTREE_TO_EV = 27.211386

_EIGEN = re.compile(r"Alpha\s+(occ\.|virt\.)\s+eigenvalues\s*--\s*(.*)")
_NUM = re.compile(r"-?\d+\.\d+")
_NEW_ATOM_ROW = re.compile(r"^\s*\d+\s+(\d+)\s+([A-Za-z]{1,2})\s")


def parse_orbital_energies(text):
    """(occ, virt) in Hartree, file order, from the LAST contiguous eigenvalue block."""
    occ_blocks, virt_blocks = [], []
    cur_occ, cur_virt = [], []
    for line in text.splitlines():
        m = _EIGEN.search(line) if "eigenvalues" in line else None
        if not m:
            if cur_occ:
                occ_blocks.append(cur_occ)
                cur_occ = []
            if cur_virt:
                virt_blocks.append(cur_virt)
                cur_virt = []
            continue
        # eigenvalues can run together ("-10.12345-10.12340"), so extract by pattern
        values = [float(v) for v in _NUM.findall(m.group(2))]
        (cur_occ if m.group(1) == "occ." else cur_virt).extend(values)
    if cur_occ:
        occ_blocks.append(cur_occ)
    if cur_virt:
        virt_blocks.append(cur_virt)
    return (occ_blocks[-1] if occ_blocks else []), (virt_blocks[-1] if virt_blocks else [])


def homo_lumo(occ, virt):
    return (occ[-1] if occ else None), (virt[0] if virt else None)


def _is_index_header(tokens):
    return 1 <= len(tokens) <= 5 and all(t.isdigit() for t in tokens)


def _is_kind_line(tokens):
    return len(tokens) >= 1 and all(t in ("O", "V") for t in tokens)


def parse_mo_composition(text, target_mo_numbers):
    """
    {mo_number: {"kind", "energy", "coeffs_by_atom": {0-based atom: [coeffs]}}}
    from the LAST 'Molecular Orbital Coefficients:' table, for 1-based MO numbers.
    Coefficients are taken by a decimal-point regex, because d/f labels such as
    '9D 0' contain an internal space that breaks positional splitting. Stops after
    the block holding the highest requested MO.
    """
    idx = text.rfind("Molecular Orbital Coefficients:")
    targets = set(target_mo_numbers)
    if idx == -1 or not targets:
        return {}
    max_target = max(targets)
    lines = text[idx:].splitlines()
    results = {mo: {"coeffs_by_atom": {}} for mo in targets}
    i, n = 1, len(lines)
    while i < n:
        tokens = lines[i].split()
        if not _is_index_header(tokens):
            i += 1
            continue
        mos = [int(t) for t in tokens]
        i += 1
        if i >= n or not _is_kind_line(lines[i].split()):
            break
        kinds = lines[i].split()
        i += 1
        if i >= n or "Eigenvalues" not in lines[i]:
            break
        energies = [float(v) for v in _NUM.findall(lines[i])]
        i += 1
        need = any(mo in targets for mo in mos)
        if need:
            for col, mo in enumerate(mos):
                if mo in targets:
                    results[mo]["kind"] = kinds[col] if col < len(kinds) else None
                    results[mo]["energy"] = energies[col] if col < len(energies) else None
        atom = None
        while i < n:
            row = lines[i]
            rt = row.split()
            if not rt or _is_index_header(rt):
                break
            m = _NEW_ATOM_ROW.match(row)
            if m:
                atom = int(m.group(1)) - 1
            elif atom is None:
                break
            if need:
                vals = _NUM.findall(row)
                for col, mo in enumerate(mos):
                    if mo in targets and col < len(vals):
                        results[mo]["coeffs_by_atom"].setdefault(atom, []).append(float(vals[col]))
            i += 1
        if mos and max(mos) >= max_target:
            break
    return results


def site_fraction(mo_data, site_atoms):
    """Share of the sum of squared coefficients that sits on site_atoms (None if empty)."""
    cba = mo_data.get("coeffs_by_atom", {})
    total = sum(c * c for cs in cba.values() for c in cs)
    if total < 1e-12:
        return None
    site = set(site_atoms)
    return sum(c * c for a, cs in cba.items() if a in site for c in cs) / total
