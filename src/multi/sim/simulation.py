"""
Toy simulation: Nickel-based superalloy microstructure evolution during heat treatment.

Models 20 microstructure evolution steps. Yield strength follows Hall-Petch (grain size).
Porosity increases with high cooling rate or improper duration; > 5% = Failed.
Schema-aligned: material_system, processing_conditions, computed_properties, microstructure.
"""
import math
from typing import Tuple

# ---------- Material system (schema: material_system) ----------
MATERIAL_NAME = "Nickel-based superalloy"
COMPOSITION = [
    {"element": "Ni", "percentage": 60},
    {"element": "Cr", "percentage": 20},
    {"element": "Co", "percentage": 10},
    {"element": "Al", "percentage": 10},
]

# ---------- Baseline constants from schema ----------
TEMPERATURE_K = 1200.0
INITIAL_GRAIN_SIZE_NM = 850.0

# ---------- Hall-Petch: sigma_y = sigma_0 + k / sqrt(grain_size_nm) ----------
SIGMA_0_MPA = 200.0
K_HALL_PETCH_MPA_NM05 = 5000.0

# ---------- Porosity / failure ----------
POROSITY_FAILURE_THRESHOLD_PERCENT = 5.0
NUM_EVOLUTION_STEPS = 20


def run_material_simulation(
    cooling_rate_K_per_min: float,
    duration_hours: float = 4.0,
    temperature_K: float = TEMPERATURE_K,
    initial_grain_size_nm: float = INITIAL_GRAIN_SIZE_NM,
    num_steps: int = NUM_EVOLUTION_STEPS,
) -> Tuple[float, bool]:
    """
    Run a toy microstructure evolution for a Nickel-based superalloy heat treatment.

    Material: Ni 60%, Cr 20%, Co 10%, Al 10%. Simulates 20 microstructure evolution
    steps. Yield strength is computed from final grain size (Hall-Petch). Porosity
    increases with high cooling rate or non-optimal duration; if it exceeds 5%,
    the material is considered failed.

    Parameters
    ----------
    cooling_rate_K_per_min : float
        Cooling rate in K/min (from processing_conditions.heat_treatment).
        Primary modifiable input. Faster cooling refines grains (higher strength)
        but can increase porosity.
    duration_hours : float
        Heat treatment duration in hours (default 4). Very short or long
        durations can increase porosity.
    temperature_K : float
        Heat treatment temperature in K (default 1200 from schema).
    initial_grain_size_nm : float
        Initial grain size in nm (default 850 from schema).
    num_steps : int
        Number of microstructure evolution steps (default 20).

    Returns
    -------
    tuple[float, bool]
        (final_yield_strength_MPa, success).
        success is False if porosity_percent > 5.0 (material "Failed").
    """
    # Microstructure state (schema: material_system.microstructure)
    grain_size_nm = initial_grain_size_nm
    porosity_percent = 1.5  # baseline from schema example

    # Evolution: faster cooling refines grains; very fast cooling or bad duration increases porosity
    cooling_factor = min(2.0, max(0.2, cooling_rate_K_per_min / 15.0))  # 15 as reference
    optimal_duration = 4.0
    duration_deviation = abs(duration_hours - optimal_duration)

    for step in range(num_steps):
        # Grain size: higher cooling rate → smaller grains (refinement)
        refinement = 1.0 + 0.03 * (step + 1) * math.log1p(cooling_factor)
        grain_size_nm = initial_grain_size_nm / refinement
        grain_size_nm = max(50.0, min(initial_grain_size_nm, grain_size_nm))

        # Porosity: increases with very high cooling (thermal stress, trapped gas) and duration mismatch
        porosity_delta_cooling = 0.08 * math.log1p(max(0, cooling_rate_K_per_min - 10))
        porosity_delta_duration = 0.02 * duration_deviation
        porosity_percent += (porosity_delta_cooling + porosity_delta_duration) / num_steps
        porosity_percent = max(0.0, min(10.0, porosity_percent))

        if porosity_percent > POROSITY_FAILURE_THRESHOLD_PERCENT:
            # Failed: still compute yield for the current grain size, then return success=False
            break

    # Yield strength from Hall-Petch (schema: computed_properties.yield_strength_MPa)
    # sigma_y = sigma_0 + k / sqrt(d_nm)
    yield_strength_MPa = SIGMA_0_MPA + K_HALL_PETCH_MPA_NM05 / math.sqrt(grain_size_nm)
    success = porosity_percent <= POROSITY_FAILURE_THRESHOLD_PERCENT

    return (yield_strength_MPa, success)


def get_material_system_dict() -> dict:
    """Return schema-shaped material_system (for reference/integration)."""
    return {
        "material_name": MATERIAL_NAME,
        "composition": COMPOSITION.copy(),
        "phase_type": "gamma/gamma-prime",
        "microstructure": {
            "grain_size_nm": INITIAL_GRAIN_SIZE_NM,
            "porosity_percent": 1.5,
            "crystal_structure": "gamma/gamma-prime",
        },
    }
