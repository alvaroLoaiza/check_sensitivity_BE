"""
Pull the handful of facts the mapping/BE checks need out of a Gaussian log:
final SCF energy, net charge, multiplicity, atom count, normal termination.

Verified against real pop=full single-point logs (29-31 MB each). Two facts
about those logs that this relies on:
  * there is no 'Recovered energy' line - the energy is the 'SCF Done' line
    (D3BJ dispersion is printed separately as 'Dispersion energy=' and is
    part of that total);
  * net charge is printed once near the top as 'Charge = -1 Multiplicity = 1'.

Also read in the same pass: the last nuclear repulsion energy (a fingerprint of
the geometry, used to prove a single point ran on the final optimized
structure) and the atomic numbers of the last orientation block (used to check
that the complex lists the monomer atoms first and the ion atoms last).
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

HARTREE_TO_KJMOL = 2625.499639

_SCF = re.compile(r"SCF Done:\s+E\(\S+\)\s*=\s*(-?\d+\.\d+)")
_CHARGE = re.compile(r"^\s*Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)")
_NATOMS = re.compile(r"^\s*NAtoms=\s*(\d+)")
_NRE = re.compile(r"nuclear repulsion energy\s+(-?\d+\.\d+)\s+Hartrees")
NRE_TOL_HA = 1e-4   # same geometry reprints the same number to ~1e-10


@dataclass
class LogInfo:
    path: str
    energy_ha: Optional[float]
    charge: Optional[int]
    multiplicity: Optional[int]
    natoms: Optional[int]
    normal_termination: bool
    nre_ha: Optional[float] = None
    atomic_numbers: Tuple[int, ...] = ()


def read_log_info(path):
    energy = charge = mult = natoms = nre = None
    znums, block, dashes = (), None, 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            # orientation block: title, dashes, 2 header lines, dashes, rows, dashes
            if block is not None:
                if line.lstrip().startswith("-----"):
                    dashes += 1
                    if dashes == 3:
                        znums, block = tuple(block), None
                elif dashes == 2:
                    tok = line.split()
                    if len(tok) == 6 and tok[1].isdigit():
                        block.append(int(tok[1]))
                continue
            if "orientation:" in line:
                block, dashes = [], 0
                continue
            # cheap substring tests first; the regexes only run on the few hits
            if "nuclear repulsion energy" in line:
                m = _NRE.search(line)
                if m:
                    nre = float(m.group(1))
            elif "SCF Done" in line:
                m = _SCF.search(line)
                if m:
                    energy = float(m.group(1))      # keep the LAST one
            elif charge is None and "Charge =" in line:
                m = _CHARGE.match(line)
                if m:
                    charge, mult = int(m.group(1)), int(m.group(2))
            elif natoms is None and "NAtoms=" in line:
                m = _NATOMS.match(line)
                if m:
                    natoms = int(m.group(1))

    # termination line is at the very end; no need to have scanned for it
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(max(0, size - 8192))
        tail = f.read().decode("utf-8", errors="ignore")
    return LogInfo(path, energy, charge, mult, natoms,
                   "Normal termination of Gaussian" in tail, nre, znums)


def read_last_nre(path):
    """Last nuclear repulsion energy in any log (e.g. the optimization log), or None."""
    nre = None
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if "nuclear repulsion energy" in line:
                m = _NRE.search(line)
                if m:
                    nre = float(m.group(1))
    return nre


def opt_logs_for(singlep_log):
    """The optimization log(s) of a single point: OPT_h/*.log next to OPT_h/singlep/."""
    return sorted(Path(singlep_log).parent.parent.glob("*.log"))


def geometry_status(singlep_nre, opt_nres):
    """'ok' if any optimization log ends on the single point's geometry,
    'no_opt_log' if there is nothing to compare with, else 'mismatch'."""
    opt_nres = [x for x in opt_nres if x is not None]
    if singlep_nre is None or not opt_nres:
        return "no_opt_log"
    if any(abs(singlep_nre - x) <= NRE_TOL_HA for x in opt_nres):
        return "ok"
    return "mismatch"


def binding_energy_kjmol(complex_info, monomer_info, ion_info):
    """BE = E_complex - E_monomer - E_ion, in kJ/mol (the definition on the slides)."""
    return (complex_info.energy_ha - monomer_info.energy_ha - ion_info.energy_ha) * HARTREE_TO_KJMOL
