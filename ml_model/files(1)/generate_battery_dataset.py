"""
generate_battery_dataset.py

Creates a SYNTHETIC battery-cycling dataset to develop and test the SOH/RUL
model against. This mimics realistic Li-ion aging patterns (capacity fade,
rising internal resistance, temperature-accelerated degradation) but it is
NOT real lab data.

Before presenting results as "trained on real data", replace this with the
NASA Prognostics Center of Excellence Battery Data Set (Saha & Goebel, 2007):
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
The loader in ml_model.py expects the same column names used here, so you
can swap in real data once it's cleaned into this shape.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

EOL_SOH_THRESHOLD = 80.0  # % SOH commonly treated as "end of useful life"


def simulate_cell(cell_id: int, n_cycles: int, ambient_temp_c: float, c_rate: float) -> pd.DataFrame:
    """Simulate one cell's full cycling history with a plausible aging curve."""
    cycle = np.arange(1, n_cycles + 1)

    # Capacity fade: gentle initial decline that accelerates with cycle count
    # and is accelerated further by higher ambient temperature and C-rate.
    temp_stress = max(0.0, (ambient_temp_c - 25.0)) * 0.00006
    rate_stress = max(0.0, (c_rate - 1.0)) * 0.00004
    fade_rate = 0.00012 + temp_stress + rate_stress

    noise = RNG.normal(0, 0.15, size=n_cycles)
    soh = 100 * np.exp(-fade_rate * cycle) + noise
    soh = np.clip(soh, 0, 100)

    # Internal resistance rises roughly inversely with SOH, plus noise.
    internal_resistance_mohm = 20 + (100 - soh) * 0.9 + RNG.normal(0, 0.5, n_cycles)

    # Sensor-level features an IoT/BMS unit would actually report per cycle.
    voltage_mean = 3.7 - (100 - soh) * 0.0025 + RNG.normal(0, 0.01, n_cycles)
    voltage_std = 0.05 + (100 - soh) * 0.0008 + RNG.normal(0, 0.003, n_cycles)
    current_mean = c_rate * 2.2 + RNG.normal(0, 0.05, n_cycles)
    temperature_mean = ambient_temp_c + current_mean * 1.8 + RNG.normal(0, 0.4, n_cycles)
    charge_time_min = 55 + (100 - soh) * 0.35 + RNG.normal(0, 1.5, n_cycles)
    discharge_capacity_ah = 2.0 * (soh / 100.0) + RNG.normal(0, 0.01, n_cycles)

    # RUL = cycles remaining until SOH crosses the end-of-life threshold.
    below = np.where(soh < EOL_SOH_THRESHOLD)[0]
    eol_cycle = below[0] if len(below) else n_cycles
    rul = np.clip(eol_cycle - cycle, 0, None)

    return pd.DataFrame({
        "cell_id": cell_id,
        "cycle_number": cycle,
        "ambient_temp_c": ambient_temp_c,
        "c_rate": c_rate,
        "voltage_mean": voltage_mean,
        "voltage_std": voltage_std,
        "current_mean": current_mean,
        "temperature_mean": temperature_mean,
        "internal_resistance_mohm": internal_resistance_mohm,
        "charge_time_min": charge_time_min,
        "discharge_capacity_ah": discharge_capacity_ah,
        "soh": soh,
        "rul": rul,
    })


def build_dataset(n_cells: int = 25, cycles_per_cell: int = 600) -> pd.DataFrame:
    frames = []
    for cell_id in range(1, n_cells + 1):
        ambient_temp_c = RNG.uniform(20, 40)   # varied storage/use conditions
        c_rate = RNG.uniform(0.5, 2.0)         # varied charge/discharge rates
        frames.append(simulate_cell(cell_id, cycles_per_cell, ambient_temp_c, c_rate))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("battery_dataset.csv", index=False)
    print(f"Wrote {len(df)} rows across {df.cell_id.nunique()} cells -> battery_dataset.csv")
    print(df.head())
