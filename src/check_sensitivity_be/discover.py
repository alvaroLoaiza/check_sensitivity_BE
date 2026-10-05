"""
Map every (monomer site, ion) in the BE table to the three single-point logs
its binding energy was built from:

    complex   <root>/<06x-08x dir>/<site>/OPT_h/singlep/*.log
    monomer   <root>/050.*/OPT_h/singlep/*.log
    ion       <root>/000.NH4_p|010.Li_p/OPT_h/singlep/*.log

Nothing here opens a log, so it is fast and does not make OneDrive download
online-only files. When a rule fails, the manifest records *what was found
instead* (e.g. the site folders that do exist), so a wrong naming rule shows
up as a readable pattern rather than a silent gap.
"""
from pathlib import Path

import pandas as pd

from .labels import ION_DIRS, bare_dir_matches, complex_dir_name, parse_label

CATEGORIES = {
    "usable": "BE present and all three logs found",
    "be_but_logs_missing": "BE present but a log was NOT found (mapping rule wrong, or local copy incomplete)",
    "logs_but_no_be": "no BE in the table but all three logs exist (BE can be computed)",
    "no_be_no_logs": "no BE and logs missing (expected: calculation never finished)",
}


def find_singlep_log(species_dir):
    """(path | None, status, detail) for <species_dir>/OPT_h/singlep/*.log"""
    species_dir = Path(species_dir)
    if not species_dir.is_dir():
        return None, "dir_not_found", species_dir.name
    sp = species_dir / "OPT_h" / "singlep"
    if not sp.is_dir():
        have = sorted(d.name for d in species_dir.iterdir() if d.is_dir())
        return None, "no_singlep_dir", "have: " + ",".join(have)
    logs = sorted(sp.glob("*.log"))
    if not logs:
        return None, "no_log", ""
    if len(logs) == 1:
        return logs[0], "ok", ""
    preferred = [p for p in logs if p.name.endswith("_singlep.log")]
    if len(preferred) == 1:
        return preferred[0], "ok", ""
    return None, "ambiguous_logs", ",".join(p.name for p in logs)


def _resolve_complex(root, lab, ion):
    cx = root / complex_dir_name(lab.mol, ion)
    if not cx.is_dir():
        return None, "dir_not_found", cx.name
    site_dir = cx / str(lab.site)
    if not site_dir.is_dir():
        have = sorted(d.name for d in cx.iterdir() if d.is_dir())
        return None, "site_dir_not_found", "%s has: %s" % (cx.name, ",".join(have))
    return find_singlep_log(site_dir)


def _resolve_bare(root, lab, entries):
    matches = [n for n in entries if bare_dir_matches(lab.mol, n)]
    if not matches:
        return None, "dir_not_found", "no 050.* dir for molecule " + lab.mol
    if len(matches) > 1:
        return None, "ambiguous_dirs", ",".join(matches)
    return find_singlep_log(root / matches[0])


def build_manifest(root, be_table):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Root folder not found: %s" % root)
    entries = sorted(d.name for d in root.iterdir() if d.is_dir())

    rows = []
    for r in be_table.itertuples(index=False):
        lab = parse_label(r.label)
        found = {
            "complex": _resolve_complex(root, lab, r.ion),
            "monomer": _resolve_bare(root, lab, entries),
            "ion": find_singlep_log(root / ION_DIRS[r.ion]),
        }
        has_be = pd.notna(r.be_kjmol)
        all_ok = all(f[1] == "ok" for f in found.values())
        if has_be and all_ok:
            cat = "usable"
        elif has_be:
            cat = "be_but_logs_missing"
        elif all_ok:
            cat = "logs_but_no_be"
        else:
            cat = "no_be_no_logs"

        row = dict(label=r.label, mol=r.mol, site=r.site, ion=r.ion,
                   be_kjmol=r.be_kjmol, has_be=has_be, category=cat)
        for kind, (path, status, detail) in found.items():
            row[kind + "_log"] = str(path) if path else ""
            row[kind + "_status"] = status
            row[kind + "_detail"] = detail
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(manifest):
    lines = ["Manifest: %d (site, ion) rows" % len(manifest), ""]
    counts = manifest["category"].value_counts()
    for cat, desc in CATEGORIES.items():
        lines.append("  %-22s %3d   %s" % (cat, counts.get(cat, 0), desc))

    bad = manifest[manifest["category"] == "be_but_logs_missing"]
    if len(bad):
        lines += ["", "Rows with a BE value but a missing log (check these first):"]
        for r in bad.itertuples():
            for kind in ("complex", "monomer", "ion"):
                status = getattr(r, kind + "_status")
                if status != "ok":
                    lines.append("  %-7s %-4s %-8s %-20s %s" % (
                        r.label, r.ion, kind, status, getattr(r, kind + "_detail")))
    return "\n".join(lines)
