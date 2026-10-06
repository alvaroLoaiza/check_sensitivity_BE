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
Site folders are named `1` or `site1` (A and B use `siteN`); the table's `L2_2` is read as site 1 of L2
(its only site folder - `verify` checks that guess numerically). TG complexes are in `060.*_p_TG_2/0/` (the `_TG_1` folders have no logs). A missing BE means
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

Opens the logs of every usable row and checks:

* charge balance and atom-count balance (complex = monomer + ion);
* atom order: the complex lists the monomer atoms first and the ion atoms last;
* normal termination of all three logs;
* each single point ran on the final optimized geometry: its nuclear repulsion energy must
  match the last one in an `OPT_h/*.log` (catches a singlep started from a stale `.chk`,
  which is what happened to Q1_1 NH4); `no_opt_log` means there was nothing to compare with;
* the BE recomputed from the logs matches the spreadsheet.

Writes `verify.csv`. Rows listed in `data/exclusions.csv` are still checked but reported
separately, not as failures.

## Excluded rows

`data/exclusions.csv` (`label, ion, reason`) lists rows left out on purpose. They stay in the
spreadsheet; `features` skips them. Currently Q1_1 (both ions): its NH4 single point ran on a
stale geometry.

## Step 3 - features

```
py -m check_sensitivity_be features
```

Writes `features.csv`: one row per usable, non-excluded (site, ion), keyed by `label, mol, site,
ion`, no file paths. `be_kjmol` is recomputed from the logs (the target).

| from | columns |
|---|---|
| bare monomer log | `monomer_charge`, `homo_mon_eV`, `homo_mon_frac_contact` (share of the monomer HOMO on the contact group's atoms; sum of squared coefficients, a relative measure) |
| bare monomer + bare ion | `gap_cross_eV` = LUMO(ion) − HOMO(monomer) |
| bare monomer, frontier orbitals | `mon_homo_m2_eV` … `mon_lumo_p2_eV` (HOMO−2 to LUMO+2), `mon_gap0/1/2_eV` = LUMO−HOMO, (LUMO+1)−(HOMO−1), (LUMO+2)−(HOMO−2), `mon_frac_contact_*` (share of each orbital on the contact group) |
| complex, geometry of the ion | `proton_transfer` (1 if an NH4⁺ H is closer to a monomer N/O than to its own N), `ion_NH_max` (longest N–H in the ion) |
| complex, NBO charges | `ion_charge_nbo` (ion fragment total), `V_r*`, `Efield_r*`, `Qnet_r*`, `n_env_r*`, `nearest_atom_*` |
| complex, geometry | `contact_group`, `contact_dist`, and `count_<type>` / `dist_<type>` for every group type |

Charge shells and group detection use the **monomer atoms only**, so the four H atoms of NH4⁺
never count as environment. Group distances are to the nearest contact atom (the O atoms of
sulfonate/phosphonate/borate, any ring carbon, otherwise the heteroatom). Any N the amine rule
does not take (bonded to S, in a ring, multiply bonded) is grouped as `N(other)`. Code:
`src/check_sensitivity_be/features/`, ported from `gaussian16-on-hpc/scripts/bin/g16*.py`.
`dist_<type>` is empty when the molecule has no group of that type.

## Step 4 - statistics

```
py -m pip install -e ".[analysis]"
py -m check_sensitivity_be analyze --plots
```

Reads `features.csv` (all rows kept, proton-transfer rows included) and uses every feature.
Features are grouped into families, because features inside one family (V/E/Q at several radii,
say) overlap too much to be separated one by one with about 55 rows per ion:

| family | features |
|---|---|
| net charge (control) | `monomer_charge` |
| (1) local charge | V, Efield, Qnet at every radius, `V_all`, `Efield_all`, `nearest_atom_charge`, `ion_charge_nbo` |
| (2) proximity | `contact_dist`, `nearest_atom_dist`, `n_env_r*`, `dist_<group>` (empty = 10 Å), `count_<group>`, contact-group type |
| (3) monomer orbitals | HOMO−2 … LUMO+2, the three gaps, their contact-group shares |
| (4) ion LUMO | `gap_cross_eV` (within one ion this is −HOMO plus a constant, so it cannot be told apart from (3)) |
| proton transfer | `proton_transfer`, `ion_NH_max` |

Writes to `analysis/` (binding energy, per ion):

| file | contents |
|---|---|
| `correlations.csv` | every feature, per ion: family, n, Spearman rho with BE, 95% CI from a bootstrap over whole molecules, partial rho with net charge removed, rho within net charge −1; `low_n` marks features present at fewer than 15 sites (e.g. most group distances) |
| `models.csv`, `family_summary.csv` | ridge regression scored leave-one-molecule-out: the full model, each family alone, the full model without each family (R² lost, and MAE added on proton-transfer rows vs the rest), checks without `ion_charge_nbo` |
| `predictions.csv` | held-out prediction for every row and model |
| `shap_features.csv`, `shap_families.csv`, `shap_summary.csv` | exact SHAP of the full linear model per row and feature (kJ/mol), summed per family, with each family's leading feature |
| `*.png` (with `--plots`) | every feature by family (correlations), family plot, parity plot, family SHAP strip coloured by the leading feature |

and to `analysis/selectivity/`, one row per site that has both ions:

| file | contents |
|---|---|
| `sites.csv` | `dBE` = BE_Li − BE_NH4 (> 0: NH4⁺ binds more strongly; the table's Diff), `affinity` = −(BE_Li + BE_NH4)/2, and the features of both complexes side by side (`@Li`, `@NH4`; monomer features once) |
| `molecules.csv` | per molecule and ion: best site, Boltzmann-weighted BE over sites (298 K), plain mean, and the resulting dBE |
| `*_dBE.*`, `*_affinity.*` | the same correlations, family models and SHAP with dBE or affinity as the target |
| `affinity_map_*.png` | \|BE_NH4\| vs \|BE_Li\| per site and per molecule (best site, Boltzmann), proton transfer marked |

The bootstrap takes about a minute; `--n-boot 500` is faster for a quick look.

The charge descriptors come from the complex, so they include the polarization and charge transfer
that binding causes: a strong correlation shows they describe binding, not that the bare monomer
predicts it.

## Notes for the analysis stage

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
