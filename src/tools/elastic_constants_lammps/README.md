# elastic_constants_lammps

Dockerised LAMMPS tool for computing elastic constants (C11, C12, C44) of
FCC/BCC metals via EAM interatomic potentials, following the
Ghafarollahi & Buehler (AtomAgents) stress-strain method.

---

## Prerequisites

- Docker installed and running on the host machine.
- EAM potential files placed in `potentials/` **before building the image** (see below).

---

## Build the image

Run from this directory (`src/tools/elastic_constants_lammps/`):

```bash
docker build -t elastic-lammps-tool:latest .
```

Expected final image size: ~1.2 GB.

The `potentials/` folder is baked into the image at build time via
`COPY potentials/ /app/potentials/`. There is no runtime volume mount.

---

## Add potential files

Place real EAM potential files in the `potentials/` subfolder **before**
running `docker build`:

```
potentials/
    Al.eam.alloy   # e.g. Mishin et al. (1999), Phys. Rev. B 59, 3393
    Cu.eam.alloy   # e.g. Mishin et al. (2001), Phys. Rev. B 63, 224106
    Ni.eam.alloy   # e.g. Mishin et al. (1999), Acta Mater. 47, 2103
    Fe.eam.fs      # e.g. Mendelev et al. (2003), Philos. Mag. 83, 3977
    W.eam.alloy    # e.g. Zhou, Johnson, Wadley (2004), Phys. Rev. B 69, 144113
    Mo.eam.alloy   # e.g. Zhou et al. (2004), Phys. Rev. B 69, 144113
```

---

## Supported elements

| Element | Structure | Trial a₀ (Å) | Mapped potential file          |
|---------|-----------|--------------|--------------------------------|
| Al      | FCC       | 4.05         | `/app/potentials/Al.eam.alloy` |
| Cu      | FCC       | 3.615        | `/app/potentials/Cu.eam.alloy` |
| Ni      | FCC       | 3.52         | `/app/potentials/Ni.eam.alloy` |
| Fe      | BCC       | 2.87         | `/app/potentials/Fe.eam.fs`    |
| W       | BCC       | 3.165        | `/app/potentials/W.eam.alloy`  |
| Mo      | BCC       | 3.147        | `/app/potentials/Mo.eam.alloy` |

The trial lattice parameter is used only as a starting geometry; the actual
equilibrium value is determined by box relaxation against the supplied
potential.

---

## Run a calculation (container directly)

```bash
docker run --rm \
    elastic-lammps-tool:latest \
    --composition Al \
    --potential /app/potentials/Al.eam.alloy \
    --supercell_size 4
```

Expected stdout (JSON):

```json
{
  "composition": "Al",
  "C11": 114.3,
  "C12": 61.8,
  "C44": 31.6,
  "runtime_seconds": 52.1,
  "status": "ok"
}
```

---

## CLI reference

| Argument           | Required | Default                           | Description |
|--------------------|----------|-----------------------------------|-------------|
| `--composition`    | Yes      | —                                 | Element symbol (Al, Cu, Ni, Fe, W, Mo) |
| `--potential`      | No       | Auto-resolved from element symbol | Absolute path to EAM/alloy potential inside container |
| `--supercell_size` | No       | 4                                 | Unit cells per axis (4 → 256 atoms for FCC) |

---

## Run from Python (host pipeline)

The simplest call — `potential` is resolved automatically from the element symbol:

```python
from src.tools.elastic_constants_lammps.host_wrapper import compute_elastic_constants_tool

results = compute_elastic_constants_tool("Al")
print(results)
# {"composition": "Al", "C11": 114.3, "C12": 61.8, "C44": 31.6,
#  "runtime_seconds": 52.1, "status": "ok"}
```

You can also pass an explicit potential path override:

```python
results = compute_elastic_constants_tool(
    composition="Al",
    potential="/app/potentials/Al_custom.eam.alloy",
    supercell_size=4,
)
```

---

## LLM tool schemas

`host_wrapper.py` exports ready-to-use schemas for both providers:

```python
from src.tools.elastic_constants_lammps.host_wrapper import (
    OPENAI_TOOL_SCHEMA,
    ANTHROPIC_TOOL_SCHEMA,
)

# OpenAI
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    tools=[OPENAI_TOOL_SCHEMA],
)

# Anthropic
response = client.messages.create(
    model="claude-sonnet-4-6",
    messages=[...],
    tools=[ANTHROPIC_TOOL_SCHEMA],
)
```

---

## Algorithm

1. **Box relaxation** — CG minimiser with `fix box/relax iso 0.0` finds the
   potential's equilibrium lattice parameter.
2. **Strain loop** — For ε ∈ {−0.01, −0.005, +0.005, +0.01} (fresh LAMMPS
   instance per strain state):
   - Apply uniaxial strain e11 (`change_box x scale`) → extract P_xx, P_yy.
   - Relax atomic positions at fixed box after each deformation.
   - C11 and C12 from linear regression of P_xx/P_yy vs ε (slope × −1 bar→GPa).
3. **C44 via [110] rotation** — Build a second box with x‖[110], y‖[−110],
   z‖[001]. Apply the same uniaxial x-strains, minimize atoms, read P_xx. Then:

   ```
   C44 = E_110 − (C11 + C12) / 2
   ```

   where E_110 is the slope of (−P_xx) vs ε in the rotated frame. This exploits
   C′₁₁ = (C11+C12)/2 + C44 for cubic symmetry and avoids the off-diagonal
   virial P_xy, which is unreliable for EAM potentials in this LAMMPS build.

---

## Files

| File                   | Purpose |
|------------------------|---------|
| `Dockerfile`           | Container definition (miniforge 24.11.0-0 + LAMMPS 2024.08.29 + numpy + scipy) |
| `elastic_tool.py`      | In-container calculation script |
| `host_wrapper.py`      | Host-side Python wrapper; exports `compute_elastic_constants_tool`, `OPENAI_TOOL_SCHEMA`, `ANTHROPIC_TOOL_SCHEMA` |
| `potentials/`          | EAM interatomic potential files — populate before building the image |
