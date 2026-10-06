"""
Functional-group classification and ion-to-group distances (port of g16geom.py).

A small, explicit, editable rule table that guesses bond order from bond
length - read and correct it, do not trust it blindly.

Two changes from the standalone script:

1. Groups are classified on the MONOMER atoms only (the complex geometry,
   ion removed). The script kept the ion in the bond graph and excluded it by
   index; dropping the ion's atoms does the same job and also covers the four
   H atoms of NH4+.

2. Distance to a group is measured to its nearest CONTACT atom, not to one
   fixed anchor. Contact atoms are the O atoms for the S/P/B oxyanion groups,
   every ring atom for aromatic rings, and the heteroatom otherwise. The script
   measured sulfonate/phosphonate/borate to the central S/P/B atom but rings to
   their nearest carbon, so the two were not comparable. Real case (A1, Li+):
   Li sits 1.95 Angstrom from a borate O, but the script ranked an aromatic
   ring (2.87, nearest C) ahead of the borate (2.94, to B).
"""
import re

from .common import adjacency, build_bonds, dist

CO_DOUBLE = (1.15, 1.26)     # carbonyl-like C=O
CO_SINGLE = (1.30, 1.48)     # ether/alcohol-like C-O
SO_RANGE = (1.40, 1.60)      # sulfonate/sulfone S-O
CC_AROMATIC = (1.33, 1.42)   # aromatic-ish C-C

CANONICAL_GROUP_TYPES = [
    "sulfonate(SO3)", "sulfonyl(SO2)", "phosphonate(PO3)", "phosphonyl(PO2)",
    "borate/boronic(BOx)", "carbonyl(C=O)", "ether(C-O-C)",
    "hydroxyl/similar(O-H)", "terminal_O(other)", "amine", "halide", "aromatic_ring",
]
_CLUSTER_TYPES = {"sulfonate(SO3)", "sulfonyl(SO2)", "phosphonate(PO3)",
                  "phosphonyl(PO2)", "borate/boronic(BOx)"}


def canonical_type(label):
    if label.startswith("amine("):
        return "amine"
    if label.startswith("halide("):
        return "halide"
    return label


def safe_name(ctype):
    """'borate/boronic(BOx)' -> 'borate_boronic_BOx' for column names."""
    return re.sub(r"[^A-Za-z0-9]+", "_", ctype).strip("_")


def _contact_atoms(label, anchor, members):
    ctype = canonical_type(label)
    if ctype in _CLUSTER_TYPES:
        return [a for a in members if a != anchor]   # the O atoms
    if ctype == "aromatic_ring":
        return list(members)
    return [anchor]


def find_all_six_membered_carbon_rings(atoms, adj):
    """Every 6-membered all-carbon ring with aromatic-range C-C bonds, deduplicated."""
    elem = [a[0] for a in atoms]
    carbons = {i for i, e in enumerate(elem) if e == "C"}
    found = set()

    def dfs(start, current, path, visited):
        if len(path) == 6:
            for j, d in adj[current]:
                if j == start and CC_AROMATIC[0] <= d <= CC_AROMATIC[1]:
                    found.add(frozenset(path))
            return
        for j, d in adj[current]:
            if j in carbons and j not in visited and CC_AROMATIC[0] <= d <= CC_AROMATIC[1]:
                dfs(start, j, path + [j], visited | {j})

    for s in sorted(carbons):
        dfs(s, s, [s], {s})
    return [tuple(sorted(r)) for r in found]


def classify_groups(atoms):
    """
    Classify functional groups of a structure that contains NO ion.
    Returns [{"label", "atoms", "anchor", "contact"}], indices into `atoms`.
    """
    adj = adjacency(atoms, build_bonds(atoms))
    elem = [a[0] for a in atoms]
    n = len(atoms)
    groups = []

    def heavy_neighbors(i):
        return [(j, d) for j, d in adj[i] if elem[j] != "H"]

    def add(label, members, anchor):
        groups.append({"label": label, "atoms": members, "anchor": anchor,
                       "contact": _contact_atoms(label, anchor, members)})

    for center, three, two in (("S", "sulfonate(SO3)", "sulfonyl(SO2)"),
                               ("P", "phosphonate(PO3)", "phosphonyl(PO2)")):
        for i in range(n):
            if elem[i] != center:
                continue
            o = [j for j, _ in heavy_neighbors(i) if elem[j] == "O"]
            if len(o) >= 3:
                add(three, [i] + o, i)
            elif len(o) == 2:
                add(two, [i] + o, i)

    for i in range(n):
        if elem[i] == "B":
            o = [j for j, _ in heavy_neighbors(i) if elem[j] == "O"]
            if len(o) >= 2:
                add("borate/boronic(BOx)", [i] + o, i)

    claimed_o = {a for g in groups for a in g["atoms"] if elem[a] == "O"}
    for i in range(n):
        if elem[i] != "O" or i in claimed_o:
            continue
        hn = heavy_neighbors(i)
        if len(hn) == 1:
            j, d = hn[0]
            if CO_DOUBLE[0] <= d <= CO_DOUBLE[1] and elem[j] == "C":
                add("carbonyl(C=O)", [i, j], i)
            else:
                hs = [k for k, _ in adj[i] if elem[k] == "H"]
                if hs:
                    add("hydroxyl/similar(O-H)", [i, j] + hs, i)
                else:
                    add("terminal_O(other)", [i, j], i)
        elif len(hn) == 2:
            add("ether(C-O-C)", [i] + [j for j, _ in hn], i)

    for i in range(n):
        if elem[i] != "N":
            continue
        nb = [j for j, _ in adj[i]]
        if all(elem[j] in ("C", "H") for j in nb) and len(nb) in (2, 3):
            n_h = sum(1 for j in nb if elem[j] == "H")
            add("amine(N%s)" % ("H" * n_h), [i] + nb, i)

    for i in range(n):
        if elem[i] in ("F", "Cl", "Br", "I"):
            heavy = [j for j, _ in adj[i] if elem[j] != "H"]
            if heavy:
                add("halide(%s)" % elem[i], [i] + heavy, i)

    for ring in find_all_six_membered_carbon_rings(atoms, adj):
        add("aromatic_ring", list(ring), ring[0])

    return groups


def group_distances(groups, atoms, probe_xyz):
    """
    Each group with its distance from the probe to its nearest contact atom,
    nearest first: [{"label", "type", "dist", "contact_atom", "atoms"}].
    `atoms` are the coordinates the group indices refer to; probe_xyz is (x, y, z).
    """
    probe = ("X",) + tuple(probe_xyz)
    out = []
    for g in groups:
        d, a = min((dist(probe, atoms[k]), k) for k in g["contact"])
        out.append({"label": g["label"], "type": canonical_type(g["label"]),
                    "dist": d, "contact_atom": a, "atoms": g["atoms"]})
    out.sort(key=lambda t: t["dist"])
    return out


def summary_by_type(pairs):
    """{ctype: (count, nearest_dist)} for the types present."""
    out = {}
    for p in pairs:
        c, d = out.get(p["type"], (0, None))
        out[p["type"]] = (c + 1, p["dist"] if d is None else min(d, p["dist"]))
    return out
