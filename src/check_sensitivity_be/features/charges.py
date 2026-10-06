"""
Local charge-environment descriptors around the ion (port of g16charge.py).

Charges: NBO natural charges (last 'Summary of Natural Population Analysis'
table), falling back to Mulliken. Descriptors at each cutoff radius, using
Coulomb decay (charge / distance, not charge * distance):

    V_r      = sum(q_i / r_i)                         e/Angstrom
    Efield_r = |sum(q_i * (r_ion - r_i) / r_i^3)|     e/Angstrom^2
    Qnet_r   = sum(q_i)  for r_i <= cutoff            e
    n_env_r  = number of atoms in the shell
    V_all, Efield_all = the same sums over ALL monomer atoms (no cutoff). A hard
               cutoff jumps whenever the shell edge crosses an atom (e.g. the S of
               a sulfonate whose O atoms are already inside), so these give one
               radius-free value per row.

Change from the standalone script: the sums run over the MONOMER atoms only.
The script summed over every atom except the ion's centre, so for NH4+ the
ion's own four H atoms (about 1.03 Angstrom from N, about +0.45 e each) were
inside every shell and dominated V, |E|, Qnet and the "nearest atom". For Li+
nothing changes, because the ion is a single atom.
"""
import math
import re

from .common import dist


def parse_nbo_charges(text):
    """
    {0-based atom index: natural charge} from the LAST NBO summary table.
    Locates the 'Atom  No    Charge' header by content, skips separator
    lines, and reads rows until one stops matching (the table closes with '=').
    """
    idx = text.rfind("Summary of Natural Population Analysis")
    if idx == -1:
        return {}
    lines = text[idx:].splitlines()
    header_i = None
    for i, line in enumerate(lines):
        if re.search(r"Atom\s+No\s+Charge", line):
            header_i = i
            break
    if header_i is None:
        return {}
    i = header_i + 1
    while i < len(lines) and re.match(r"^\s*-+\s*$", lines[i]):
        i += 1
    row = re.compile(r"^\s*([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)")
    charges = {}
    while i < len(lines):
        m = row.match(lines[i])
        if not m:
            break
        charges[int(m.group(2)) - 1] = float(m.group(3))
        i += 1
    return charges


def parse_mulliken_charges(text):
    blocks = list(re.finditer(
        r"Mulliken charges:\s*\n\s*\d+\s*\n(.*?)\n\s*Sum of Mulliken charges", text, re.DOTALL))
    if not blocks:
        return {}
    charges = {}
    for line in blocks[-1].group(1).splitlines():
        tokens = line.split()
        if len(tokens) != 3:
            continue
        try:
            charges[int(tokens[0]) - 1] = float(tokens[2])
        except ValueError:
            continue
    return charges


def get_charges(text):
    """(charges, source) with source 'NBO' or 'Mulliken'; ({}, None) if neither is present."""
    nbo = parse_nbo_charges(text)
    if nbo:
        return nbo, "NBO"
    mull = parse_mulliken_charges(text)
    if mull:
        return mull, "Mulliken"
    return {}, None


def compute_descriptors(atoms, charges, probe_idx, radii, env_indices):
    """Shell descriptors around atoms[probe_idx], summing only over env_indices."""
    px, py, pz = atoms[probe_idx][1:4]
    env = []
    for i in env_indices:
        if i == probe_idx or i not in charges:
            continue
        r = dist(atoms[probe_idx], atoms[i])
        if r > 1e-6:
            env.append((i, atoms[i][0], r, charges[i]))

    out = {}
    for radius in radii:
        shell = [t for t in env if t[2] <= radius]
        ex = sum(q * (px - atoms[i][1]) / r ** 3 for i, _, r, q in shell)
        ey = sum(q * (py - atoms[i][2]) / r ** 3 for i, _, r, q in shell)
        ez = sum(q * (pz - atoms[i][3]) / r ** 3 for i, _, r, q in shell)
        tag = "%g" % radius
        out["V_r" + tag] = sum(q / r for _, _, r, q in shell)
        out["Efield_r" + tag] = math.sqrt(ex * ex + ey * ey + ez * ez)
        out["Qnet_r" + tag] = sum(q for _, _, _, q in shell)
        out["n_env_r" + tag] = len(shell)

    ex = sum(q * (px - atoms[i][1]) / r ** 3 for i, _, r, q in env)
    ey = sum(q * (py - atoms[i][2]) / r ** 3 for i, _, r, q in env)
    ez = sum(q * (pz - atoms[i][3]) / r ** 3 for i, _, r, q in env)
    out["V_all"] = sum(q / r for _, _, r, q in env)
    out["Efield_all"] = math.sqrt(ex * ex + ey * ey + ez * ez)

    if env:
        _, elem, r, q = min(env, key=lambda t: t[2])
        out["nearest_atom_elem"] = elem
        out["nearest_atom_dist"] = r
        out["nearest_atom_charge"] = q
    return out
