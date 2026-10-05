"""
BE-table labels -> folder names in the LONI_work_folder tree.

Label grammar (from all_BE_values.xlsx):
    A1, C4, K5        molecule letter + site number
    L1_1, M2_2, P1_3  molecule name that itself contains a digit (L1, L2, M1,
                      M2, P1, P2, Q1, Q2), then '_' + site number
    TG, CH3SO3        single-site molecules, no site in the label (their
                      complexes sit in a '0' site folder)

Folder rules (as described by the data's author; edit here if they change):
    ions            000.NH4_p , 010.Li_p
    bare monomers   050.*          (suffix varies: _m, _2m, none)
    complexes       060-080.*      <dir>/<site>/OPT_h/singlep/*.log
                    where <site> is '1' or 'site1' (A and B use 'siteN')
    A = C22H24BO3,  B = C11H9N2O2S  (named by formula, not mol_<X>)
    TG complexes live in 060.<ion>_p_TG_2/0/ (the *_TG_1 dirs have no logs)
"""
import re
from dataclasses import dataclass

ION_DIRS = {"NH4": "000.NH4_p", "Li": "010.Li_p"}
ION_COLUMNS = {"BE_NH4+": "NH4", "BE_Li+": "Li"}

# Molecules named by formula or special-cased. {ion} is "Li" or "NH4".
NAMED = {
    "A": dict(bare="050.C22H24BO3_m", cx="070.{ion}_C22H24BO3"),
    "B": dict(bare="050.C11H9N2O2S_m", cx="075.{ion}_C11H9N2O2S"),
    "CH3SO3": dict(bare="050.CH3SO3_m", cx="060.{ion}_CH3SO3"),
    "TG": dict(bare="050.TG", cx="060.{ion}_p_TG_2"),
}
SINGLE_SITE = {"TG", "CH3SO3"}

# Table labels that do not follow the folder numbering. The BE table calls the
# single site of molecule L2 "L2_2", but 080.*_mol_L2 only has a '1' site folder
# (and the slides show one site for L2). This is an assumption: `verify`
# recomputes BE from the logs, so the table value for L2_2 will either match
# the L2 site-1 logs or expose the guess as wrong.
LABEL_ALIASES = {"L2_2": ("L2", 1)}

_TWO_PART = re.compile(r"^(?P<mol>[A-Z]\d)_(?P<site>\d+)$")
_ONE_PART = re.compile(r"^(?P<mol>[A-Z])(?P<site>\d+)$")


@dataclass(frozen=True)
class Label:
    raw: str
    mol: str   # 'A', 'C', 'L1', 'TG', 'CH3SO3'
    site: int  # 0 for single-site molecules


def parse_label(raw):
    raw = str(raw).strip()
    if raw in LABEL_ALIASES:
        mol, site = LABEL_ALIASES[raw]
        return Label(raw, mol, site)
    if raw in SINGLE_SITE:
        return Label(raw, raw, 0)
    m = _TWO_PART.match(raw) or _ONE_PART.match(raw)
    if not m:
        raise ValueError("Unrecognized monomer label: %r" % raw)
    return Label(raw, m.group("mol"), int(m.group("site")))


def complex_dir_name(mol, ion):
    if mol in NAMED:
        return NAMED[mol]["cx"].format(ion=ion)
    return "080.%s_mol_%s" % (ion, mol)


def bare_dir_matches(mol, dirname):
    """True if `dirname` (a name directly under the root) is mol's bare-monomer folder."""
    if mol in NAMED:
        return dirname == NAMED[mol]["bare"]
    return re.match(r"^050\.mol_%s(_[A-Za-z0-9]+)?$" % re.escape(mol), dirname) is not None
