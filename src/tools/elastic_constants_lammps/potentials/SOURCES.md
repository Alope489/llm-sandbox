# EAM Potential File Sources

All files in this directory are downloaded from the NIST Interatomic Potentials Repository
(https://www.ctcms.nist.gov/potentials/) and renamed to canonical project names.
They are committed as static reference data and do not need to be updated on each run.

| Canonical filename | Original NIST filename | NIST URL | Paper reference | Downloaded |
|--------------------|----------------------|----------|-----------------|------------|
| `Al.eam.alloy` | `Al99.eam.alloy` | https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Al/2/Al99.eam.alloy | Y. Mishin, D. Farkas, M.J. Mehl, D.A. Papaconstantopoulos, *Phys. Rev. B* **59**, 3393 (1999) | 2026-02-27 |
| `Cu.eam.alloy` | `Cu01.eam.alloy` | https://www.ctcms.nist.gov/potentials/Download/2001--Mishin-Y-Mehl-M-J-Papaconstantopoulos-D-A-et-al--Cu-1/2/Cu01.eam.alloy | Y. Mishin, M.J. Mehl, D.A. Papaconstantopoulos, A.F. Voter, J.D. Kress, *Phys. Rev. B* **63**, 224106 (2001) | 2026-02-27 |
| `Ni.eam.alloy` | `Ni99.eam.alloy` | https://www.ctcms.nist.gov/potentials/Download/1999--Mishin-Y-Farkas-D-Mehl-M-J-Papaconstantopoulos-D-A--Ni/2/Ni99.eam.alloy | Y. Mishin, D. Farkas, M.J. Mehl, D.A. Papaconstantopoulos, *Phys. Rev. B* **59**, 3393 (1999) | 2026-02-27 |
| `Fe.eam.fs` | `Fe_2.eam.fs` | https://www.ctcms.nist.gov/potentials/Download/2003--Mendelev-M-I-Han-S-Srolovitz-D-J-et-al--Fe-2/3/Fe_2.eam.fs | M.I. Mendelev, S. Han, D.J. Srolovitz, G.J. Ackland, D.Y. Sun, M. Asta, *Philos. Mag.* **83**, 3977 (2003) | 2026-02-27 |
| `W.eam.alloy` | `W_Zhou04.eam.alloy` | https://www.ctcms.nist.gov/potentials/Download/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--W/4/W_Zhou04.eam.alloy | X.W. Zhou, R.A. Johnson, H.N.G. Wadley, *Phys. Rev. B* **69**, 144113 (2004) | 2026-02-27 |
| `Mo.eam.alloy` | `Mo_Zhou04.eam.alloy` | https://www.ctcms.nist.gov/potentials/Download/2004--Zhou-X-W-Johnson-R-A-Wadley-H-N-G--Mo/4/Mo_Zhou04.eam.alloy | X.W. Zhou, R.A. Johnson, H.N.G. Wadley, *Phys. Rev. B* **69**, 144113 (2004) | 2026-02-27 |

## Notes

- `Al.eam.alloy`, `Cu.eam.alloy`, `Ni.eam.alloy` use LAMMPS `pair_style eam/alloy`.
- `Fe.eam.fs` uses LAMMPS `pair_style eam/fs` (Finnis-Sinclair format); `elastic_tool.py`
  detects this automatically from the `.eam.fs` extension.
- `W.eam.alloy` and `Mo.eam.alloy` use LAMMPS `pair_style eam/alloy`.
- The Mendelev Fe potential (potential #2 from the 2003 paper) is used; it provides
  better liquid-iron properties but also good crystalline elastic constants.
- The Zhou 2004 W and Mo potentials are the corrected 2018 retabulation (ipr2) that fixes
  spurious fluctuations in the original tabulated functions.
