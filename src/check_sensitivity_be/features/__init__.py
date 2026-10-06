"""
Feature extraction from Gaussian single-point logs (ported from the standalone
scripts in gaussian16-on-hpc/scripts/bin: g16features_common, g16charge,
g16geom, g16orbitals, g16extract_all).

    common    geometry parsing, bond graph, distances
    charges   NBO (fallback Mulliken) charges and the Coulomb-decay descriptors
    groups    functional-group classification and ion-to-group distances
    orbitals  orbital energies and site-projected MO composition
    extract   one feature row per (site, ion), from the complex + bare monomer + bare ion
"""
