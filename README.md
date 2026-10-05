# check_sensitivity_BE

Do local **charge**, **functional-group proximity** and **frontier-orbital** descriptors
explain the Li⁺ / NH4⁺ binding energies (BE) of the candidate monomer sites?

`BE = E_complex − E_monomer − E_ion` (kJ/mol), all from PBE0-D3BJ/def2TZVP single points
(`pop=(full,nbo)`) in `OPT_h/singlep/`.

## Data layout (LONI_work_folder, copied from `/work/alvaro/gaussian-from-home`)

| what | folder | notes |
|---|---|---|
| cations | `000.NH4_p`, `010.Li_p` | `OPT_h/singlep/*.log` |
| bare monomers | `050.*` | suffix varies (`_m`, `_2m`, none) |
| complexes | `060`–`080.*` / `<site>` | `OPT_h/singlep/*.log` |

Labels in `data/all_BE_values.xlsx`: `A2` = molecule A, site 2; molecules that already contain a
digit get a second one (`M1_2` = molecule M1, site 2). `A` = C22H24BO3, `B` = C11H9N2O2S.
TG complexes are in `060.*_p_TG_2/0/` (the `_TG_1` folders have no logs). A missing BE means
the single-point log did not exist when the table was made. All rules live in
`src/check_sensitivity_be/labels.py`.

## Install

```
py -m pip install -e .            # numpy, pandas, openpyxl
py -m pip install -e .[analysis]  # later: scikit-learn, shap, matplotlib
py -m pip install -e .[dev]       # pytest
```

## Step 1 - map every BE value to its three logs

```
py -m check_sensitivity_be manifest --root "C:\Users\zheru\OneDrive - Louisiana State University\Desktop\P26\gaussian\git\gaussian16-on-hpc\jobs\runs\LONI_work_folder"
```

Writes `manifest.csv` and prints how many rows are `usable`, plus every row that has a BE
value but a missing log (with what *was* found instead). It only lists folders, so it is fast
and does not make OneDrive download anything.

## Step 2 - prove the mapping is right

```
py -m check_sensitivity_be verify
```

Opens the logs of every usable row and checks: charge balance and atom-count balance
(complex = monomer + ion), normal termination, and that the BE recomputed from the logs
matches the spreadsheet. Writes `verify.csv`. Any row that fails a check is listed - those
are either a wrong folder mapping or a stale spreadsheet value.

## Notes for the analysis stages (not built yet)

* **Net charge dominates BE** (dianion monomers reach −118 to −172 kJ/mol; neutral P1/Q1 sit near
  −10 to −28). Monomer net charge is read from the log (`Charge = -1`), not from folder names,
  and must be a feature or a stratification variable.
* **Sites of one molecule share monomer-level features**, so cross-validation must be grouped by
  molecule (23 molecules, 66 sites, 122 BE values).
* **Ion LUMO is a per-ion constant** - it equals "which ion" in pooled data and has zero variance
  within one ion. Its useful form is the cross-fragment gap `LUMO_ion − HOMO_monomer`.
* Bare-monomer atoms map 1:1 to the first `n_monomer` atoms of the complex (ion atoms are last),
  which is what lets the site atoms found in the complex be used to project the monomer's HOMO.

Raw Gaussian output (`*.log`, `*.chk`) is git-ignored; it stays on OneDrive/LONI.
